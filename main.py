import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Define dataset and data loader
train_transforms = transforms.Compose([
    transforms.ToTensor()
])

train_data = datasets.MNIST(root="data", train=True, download=True, transform=train_transforms)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

# Define model
class ImageClassifier(nn.Module):
    def __init__(self):
        super(ImageClassifier, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU()
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*22*22, 10)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Define loss function and optimizer
model = ImageClassifier()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Train model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

for epoch in range(10):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)

    print(f"Epoch {epoch+1} loss: {running_loss/len(train_loader.dataset):.4f}")

# Save model
torch.save(model.state_dict(), 'model.pth')

# Load model
model.load_state_dict(torch.load('model.pth'))

# Inference
img = Image.open('img_3.jpg')
img_tensor = train_transforms(img).unsqueeze(0).to(device)
output = model(img_tensor)
pred = torch.argmax(output, dim=1)
print(f"Prediction: {pred.item()}")
