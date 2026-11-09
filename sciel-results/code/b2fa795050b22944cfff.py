"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.linalg import solve


def fit(X_train, y_train):
    n, d = X_train.shape
    
    # Define frequency grid
    # We want to capture functions on domains up to [1, 10].
    # A frequency k corresponds to a period of 2*pi/k.
    # If k=1, period ~6.28. If k=5, period ~1.25.
    # Let's use integer frequencies from 1 to K_max.
    K_max = 10
    freqs = np.arange(1, K_max + 1)
    
    # For cross terms, use a smaller set of frequencies to control feature explosion
    K_cross = 4
    cross_freqs = np.arange(1, K_cross + 1)
    
    features = []
    
    # 1. Bias
    features.append(np.ones(n))
    
    # 2. Single dimension terms: sin(k * x_j), cos(k * x_j)
    for j in range(d):
        x_j = X_train[:, j]
        for k in freqs:
            features.append(np.sin(k * x_j))
            features.append(np.cos(k * x_j))
            
    # 3. Cross terms for pairs of dimensions
    # To keep feature count low, we only include cross terms for pairs (i, j) with i < j
    # and only for the smaller set of cross_freqs
    for i in range(d):
        for j in range(i + 1, d):
            x_i = X_train[:, i]
            x_j = X_train[:, j]
            # Precompute sines and cosines for x_i and x_j with cross_freqs
            sin_i = np.column_stack([np.sin(k * x_i) for k in cross_freqs])
            cos_i = np.column_stack([np.cos(k * x_i) for k in cross_freqs])
            sin_j = np.column_stack([np.sin(k * x_j) for k in cross_freqs])
            cos_j = np.column_stack([np.cos(k * x_j) for k in cross_freqs])
            
            # Products:
            # sin_i * sin_j
            features.extend(np.multiply.outer(sin_i, sin_j).T.flatten()) # This is wrong shape, need to be careful
            # Let's do it element-wise properly
            for a in range(len(cross_freqs)):
                for b in range(len(cross_freqs)):
                    features.append(sin_i[:, a] * sin_j[:, b])
                    features.append(sin_i[:, a] * cos_j[:, b])
                    features.append(cos_i[:, a] * sin_j[:, b])
                    features.append(cos_i[:, a] * cos_j[:, b])

    X_feat = np.column_stack(features)
    
    # Regularized Least Squares
    # Alpha needs to be tuned. With ~150-200 features and 200 points, 
    # a small alpha is good.
    alpha = 1e-2
    n_feat = X_feat.shape[1]
    
    # Solve (X^T X + alpha * I) w = X^T y
    XtX = X_feat.T @ X_feat
    Xty = X_feat.T @ y_train
    
    # Add small ridge to ensure invertibility
    A = XtX + alpha * np.eye(n_feat)
    w = solve(A, Xty, assume_a='pos')
    
    def predict(X):
        n_test = X.shape[0]
        features_test = []
        
        features_test.append(np.ones(n_test))
        
        for j in range(d):
            x_j = X[:, j]
            for k in freqs:
                features_test.append(np.sin(k * x_j))
                features_test.append(np.cos(k * x_j))
                
        for i in range(d):
            for j in range(i + 1, d):
                x_i = X[:, i]
                x_j = X[:, j]
                sin_i = np.column_stack([np.sin(k * x_i) for k in cross_freqs])
                cos_i = np.column_stack([np.cos(k * x_i) for k in cross_freqs])
                sin_j = np.column_stack([np.sin(k * x_j) for k in cross_freqs])
                cos_j = np.column_stack([np.cos(k * x_j) for k in cross_freqs])
                
                for a in range(len(cross_freqs)):
                    for b in range(len(cross_freqs)):
                        features_test.append(sin_i[:, a] * sin_j[:, b])
                        features_test.append(sin_i[:, a] * cos_j[:, b])
                        features_test.append(cos_i[:, a] * sin_j[:, b])
                        features_test.append(cos_i[:, a] * cos_j[:, b])

        X_feat_test = np.column_stack(features_test)
        return X_feat_test @ w
        
    return predict
# EVOLVE-BLOCK-END
