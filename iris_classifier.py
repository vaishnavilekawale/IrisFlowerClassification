# ================================
# Iris Flower Classification
# ================================

# 🔹 Step 1: Import Required Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# 🔹 Step 2: Load the Iris Dataset
iris = load_iris()

# Create DataFrame for better visualization
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

# 🔹 Step 3: Display Dataset Summary
print("First 5 rows of the Iris dataset:\n")
print(df.head())

# 🔹 Step 4: Visualize the Dataset (Pairplot)
sns.pairplot(df, hue='species')
plt.suptitle("Iris Dataset Pairplot", y=1.02)  # 🔧 No emojis used here
plt.show()

# 🔹 Step 5: Prepare Features (X) and Labels (y)
X = iris.data
y = iris.target

# 🔹 Step 6: Split into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 🔹 Step 7: Standardize the Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 🔹 Step 8: Initialize Classifiers
models = {
    'Logistic Regression': LogisticRegression(max_iter=200),  # Increased max_iter for stability
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}

# 🔹 Step 9: Train & Evaluate Each Model
for name, model in models.items():
    print(f"\n==============================")
    print(f"Model: {name}")
    print(f"==============================")
    
    model.fit(X_train, y_train)              # Train
    y_pred = model.predict(X_test)           # Predict
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=iris.target_names)

    print(f"Accuracy: {acc:.2f}")
    print("\nClassification Report:")
    print(report)
    print("Confusion Matrix:")
    print(cm)
