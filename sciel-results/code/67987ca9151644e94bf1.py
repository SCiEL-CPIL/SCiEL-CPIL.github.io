import numpy as np
import time
from scipy.linalg import solve

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 290.0  # Safety margin from 300s limit
    
    n, d = X_train.shape
    
    # Normalize inputs to [0, 1] for feature generation stability
    min_vals = X_train.min(axis=0)
    max_vals = X_train.max(axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0
    X_norm = (X_train - min_vals) / ranges
    
    # Generate Basis Features
    # We want to capture polynomials, trig, exp, log, rational
    features = [np.ones(n)]
    
    # Polynomials up to degree 3
    for i in range(d):
        features.append(X_norm[:, i])
    for i in range(d):
        for j in range(i, d):
            features.append(X_norm[:, i] * X_norm[:, j])
    for i in range(d):
        features.append(X_norm[:, i] ** 3)
        # Also add x^2 * x_j for i != j? Maybe too many. Let's stick to pure cubics and quadratics for now.
        # Actually, full degree 3 includes x_i^2 x_j.
    for i in range(d):
        for j in range(i + 1, d):
            features.append(X_norm[:, i] ** 2 * X_norm[:, j])
            features.append(X_norm[:, i] * X_norm[:, j] ** 2)
            
    # Trigonometric
    for i in range(d):
        features.append(np.sin(X_norm[:, i]))
        features.append(np.cos(X_norm[:, i]))
        
    # Exponential and Logarithmic (using normalized inputs to keep scale manageable)
    for i in range(d):
        features.append(np.exp(-X_norm[:, i]))
        features.append(np.log(X_norm[:, i] + 0.1)) # +0.1 to avoid log(0) if min is 0, but min is 1 in raw, so norm min is 0.
        features.append(1.0 / (X_norm[:, i] + 0.1))
        
    # Combine features
    X_feat = np.column_stack(features)
    
    # Handle NaNs or Infs just in case
    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Split for validation
    # Use 80/20 split
    n_val = int(n * 0.2)
    val_idx = np.arange(n - n_val, n)
    train_idx = np.arange(0, n - n_val)
    
    X_tr_feat = X_feat[train_idx]
    y_tr = y_train[train_idx]
    X_val_feat = X_feat[val_idx]
    y_val = y_train[val_idx]
    
    best_model_type = None
    best_params = None
    best_val_mse = np.inf
    
    # 1. Try Ridge Regression on the basis
    # We can try a few alpha values
    alphas_ridge = [1e-6, 1e-4, 1e-2, 1e0]
    
    for alpha in alphas_ridge:
        try:
            # Solve (X^T X + alpha I) w = X^T y
            XtX = X_tr_feat.T @ X_tr_feat
            Xty = X_tr_feat.T @ y_tr
            w = solve(XtX + alpha * np.eye(XtX.shape[0]), Xty, assume_a='pos')
            
            y_pred_val = X_val_feat @ w
            mse = np.mean((y_pred_val - y_val) ** 2)
            
            if mse < best_val_mse:
                best_val_mse = mse
                best_model_type = 'ridge'
                best_params = {'alpha': alpha}
        except:
            pass
            
    # 2. Try GP with a few hyperparameters
    # Use normalized X
    X_tr_norm = X_norm[train_idx]
    X_val_norm = X_norm[val_idx]
    
    def compute_kernel(X1, X2, gamma, length_scale=1.0):
        # RBF Kernel
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * (X1 @ X2.T)
        sq_dists = np.maximum(sq_dists, 0)
        K_rbf = np.exp(-gamma * sq_dists)
        
        # Add a small constant kernel for bias? Or just rely on y-centering?
        # Let's add a small white noise kernel for numerical stability
        return K_rbf + 1e-6 * np.eye(len(X1)) if len(X1) == len(X2) else K_rbf

    gammas = [1.0, 5.0, 10.0, 20.0, 50.0]
    
    for gamma in gammas:
        try:
            K_tr = compute_kernel(X_tr_norm, X_tr_norm, gamma)
            K_val = compute_kernel(X_val_norm, X_tr_norm, gamma)
            
            # Solve
            w = solve(K_tr, y_tr, assume_a='pos')
            y_pred_val = K_val @ w
            mse = np.mean((y_pred_val - y_val) ** 2)
            
            if mse < best_val_mse:
                best_val_mse = mse
                best_model_type = 'gp'
                best_params = {'gamma': gamma}
        except:
            pass

    # If no model was found (should not happen), default to GP with gamma=10
    if best_model_type is None:
        best_model_type = 'gp'
        best_params = {'gamma': 10.0}
        
    # Final Fit on all data with best parameters
    if best_model_type == 'ridge':
        alpha = best_params['alpha']
        XtX = X_feat.T @ X_feat
        Xty = X_feat.T @ y_train
        w = solve(XtX + alpha * np.eye(XtX.shape[0]), Xty, assume_a='pos')
        
        def predict(X):
            X_norm_test = (X - min_vals) / ranges
            # Regenerate features for test set
            feats_test = [np.ones(len(X))]
            for i in range(d):
                feats_test.append(X_norm_test[:, i])
            for i in range(d):
                for j in range(i, d):
                    feats_test.append(X_norm_test[:, i] * X_norm_test[:, j])
            for i in range(d):
                feats_test.append(X_norm_test[:, i] ** 3)
            for i in range(d):
                for j in range(i + 1, d):
                    feats_test.append(X_norm_test[:, i] ** 2 * X_norm_test[:, j])
                    feats_test.append(X_norm_test[:, i] * X_norm_test[:, j] ** 2)
            for i in range(d):
                feats_test.append(np.sin(X_norm_test[:, i]))
                feats_test.append(np.cos(X_norm_test[:, i]))
            for i in range(d):
                feats_test.append(np.exp(-X_norm_test[:, i]))
                feats_test.append(np.log(X_norm_test[:, i] + 0.1))
                feats_test.append(1.0 / (X_norm_test[:, i] + 0.1))
                
            X_feat_test = np.column_stack(feats_test)
            X_feat_test = np.nan_to_num(X_feat_test, nan=0.0, posinf=1e6, neginf=-1e6)
            return X_feat_test @ w

    else: # GP
        gamma = best_params['gamma']
        K_train = compute_kernel(X_norm, X_norm, gamma)
        w = solve(K_train, y_train, assume_a='pos')
        
        def predict(X):
            X_norm_test = (X - min_vals) / ranges
            K_test = compute_kernel(X_norm_test, X_norm, gamma)
            return K_test @ w

    return predict
