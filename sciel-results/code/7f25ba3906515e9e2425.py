"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def fit(X_train, y_train):
    d = X_train.shape[1]
    n = len(X_train)
    
    # Normalize inputs to [0, 1]
    X_min = X_train.min(axis=0, keepdims=True)
    X_max = X_train.max(axis=0, keepdims=True)
    X_range = X_max - X_min
    X_range[X_range < 1e-10] = 1.0
    
    def normalize(X):
        return (X - X_min) / X_range
    
    Xn_train = normalize(X_train)
    
    # Approach: Hybrid Feature Expansion (Polynomial + Fourier) + Ridge Regression
    # This is distinct from Kernel Ridge Regression as it uses a fixed finite basis.
    
    def build_features(X):
        features = [np.ones((X.shape[0], 1))]
        
        # 1. Linear terms
        features.append(X)
        
        # 2. Polynomial terms (degree 2 and 3)
        # We generate all monomials of degree 2 and 3
        # For d=1: x^2, x^3
        # For d=2: x1^2, x2^2, x1x2, x1^3, x2^3, x1^2x2, x1x2^2
        # This can explode, so we limit to degree 2 for d>2, or use specific interactions.
        # Let's use a safe set: Squares and Pairwise Products (degree 2)
        for i in range(d):
            features.append(X[:, i:i+1] ** 2)
            for j in range(i+1, d):
                features.append(X[:, i:i+1] * X[:, j:j+1])
                
        # Add degree 3 terms only if d is small (<=3) to avoid dimensionality explosion
        if d <= 3:
            for i in range(d):
                features.append(X[:, i:i+1] ** 3)
                for j in range(i+1, d):
                    features.append(X[:, i:i+1] ** 2 * X[:, j:j+1])
                    features.append(X[:, i:i+1] * X[:, j:j+1] ** 2)
            if d == 3:
                 # x1^2 x2 x3, etc.
                 features.append(X[:, 0:1] ** 2 * X[:, 1:2] * X[:, 2:3])
                 features.append(X[:, 0:1] * X[:, 1:2] ** 2 * X[:, 2:3])
                 features.append(X[:, 0:1] * X[:, 1:2] * X[:, 2:3] ** 2)
                 features.append(X[:, 0:1] * X[:, 1:2] * X[:, 2:3]) # Already in degree 2? No, x1x2 is deg 2. x1x2x3 is deg 3.
        
        # 3. Fourier terms (Sine/Cosine) to capture periodicity
        # Frequencies: 1, 2
        for k in [1, 2]:
            for i in range(d):
                features.append(np.sin(2 * np.pi * k * X[:, i:i+1]))
                features.append(np.cos(2 * np.pi * k * X[:, i:i+1]))
                
        return np.hstack(features)
    
    # Build training features
    X_feat_train = build_features(Xn_train)
    
    # Cross-validation to find best alpha (Ridge parameter)
    # We also consider if we should drop some features? No, let's just tune alpha.
    alphas = [1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
    
    n_folds = 5
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    best_alpha = 1e-2
    best_cv_mse = np.inf
    
    for alpha in alphas:
        cv_mses = []
        for k in range(n_folds):
            val_idx = folds[k]
            train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != k])
            
            X_tr = X_feat_train[train_idx]
            y_tr = y_train[train_idx]
            X_val = X_feat_train[val_idx]
            y_val = y_train[val_idx]
            
            # Solve Ridge Regression: (X^T X + alpha I) w = X^T y
            # Using lstsq for stability
            try:
                w, _, _, _ = np.linalg.lstsq(X_tr.T @ X_tr + alpha * np.eye(X_tr.shape[1]), X_tr.T @ y_tr, rcond=None)
            except np.linalg.LinAlgError:
                w = np.zeros(X_tr.shape[1])
            
            pred = X_val @ w
            mse = np.mean((pred - y_val) ** 2)
            cv_mses.append(mse)
        
        mean_cv_mse = np.mean(cv_mses)
        if mean_cv_mse < best_cv_mse:
            best_cv_mse = mean_cv_mse
            best_alpha = alpha
            
    # Final fit
    X_full_feat = X_feat_train
    try:
        w_final, _, _, _ = np.linalg.lstsq(X_full_feat.T @ X_full_feat + best_alpha * np.eye(X_full_feat.shape[1]), X_full_feat.T @ y_train, rcond=None)
    except np.linalg.LinAlgError:
        w_final = np.zeros(X_full_feat.shape[1])
        
    def predict(X):
        Xn = normalize(X)
        X_feat_test = build_features(Xn)
        return X_feat_test @ w_final
    
    return predict
# EVOLVE-BLOCK-END
