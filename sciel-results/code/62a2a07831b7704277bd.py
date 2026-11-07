import numpy as np
import time
from scipy.linalg import solve
from scipy.optimize import minimize
import warnings

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 280.0  # Safety margin from 300s limit
    
    n, d = X_train.shape
    
    # Normalize inputs to [0, 1] for feature generation stability
    min_vals = X_train.min(axis=0)
    max_vals = X_train.max(axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0
    X_norm = (X_train - min_vals) / ranges
    
    # Generate a rich set of basis features
    def generate_features(X):
        n = X.shape[0]
        d = X.shape[1]
        feats = [np.ones(n)]
        
        # Linear terms
        for i in range(d):
            feats.append(X[:, i])
        
        # Quadratic terms
        for i in range(d):
            for j in range(i, d):
                feats.append(X[:, i] * X[:, j])
        
        # Cubic terms (pure and mixed)
        for i in range(d):
            for j in range(d):
                for k in range(d):
                    if i <= j <= k:
                        feats.append(X[:, i] * X[:, j] * X[:, k])
        
        # Trigonometric functions
        for i in range(d):
            feats.append(np.sin(X[:, i]))
            feats.append(np.cos(X[:, i]))
            # Second harmonics
            feats.append(np.sin(2 * np.pi * X[:, i]))
            feats.append(np.cos(2 * np.pi * X[:, i]))
        
        # Exponential and log functions
        for i in range(d):
            feats.append(np.exp(-2 * X[:, i]))
            feats.append(np.exp(-X[:, i]))
            feats.append(np.log(X[:, i] + 1e-6))
            feats.append(1.0 / (X[:, i] + 1e-6))
        
        # Pairwise interactions with trig
        if d >= 2:
            for i in range(d):
                for j in range(i + 1, d):
                    feats.append(np.sin(X[:, i]) * np.cos(X[:, j]))
                    feats.append(np.sin(X[:, i]) * np.sin(X[:, j]))
        
        X_feat = np.column_stack(feats)
        X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=1e6, neginf=-1e6)
        return X_feat

    # We should also generate features based on RAW inputs for exp/log to be accurate
    def generate_features_raw(X):
        n = X.shape[0]
        d = X.shape[1]
        feats = [np.ones(n)]
        for i in range(d):
            feats.append(X[:, i])
        for i in range(d):
            feats.append(np.exp(-X[:, i] / 10.0))  # scaled exp
            feats.append(np.log(X[:, i]))
            feats.append(1.0 / X[:, i])
            feats.append(np.sin(X[:, i]))
            feats.append(np.cos(X[:, i]))
        X_feat = np.column_stack(feats)
        X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=1e6, neginf=-1e6)
        return X_feat

    X_feat_norm = generate_features(X_norm)
    X_feat_raw = generate_features_raw(X_train)
    X_feat_all = np.hstack([X_feat_norm, X_feat_raw])
    
    # Center y for numerical stability
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    if y_std == 0:
        y_std = 1.0
    y_centered = y_train - y_mean

    # Split for validation (last 20% for simplicity)
    n_val = max(10, int(n * 0.2))
    val_idx = np.arange(n - n_val, n)
    train_idx = np.arange(0, n - n_val)
    
    X_tr = X_feat_all[train_idx]
    y_tr = y_centered[train_idx]
    X_val = X_feat_all[val_idx]
    y_val = y_centered[val_idx]
    
    best_model = None
    best_val_mse = np.inf
    best_type = None
    
    # 1. Ridge Regression with a range of alphas
    # Number of features is likely much larger than 200, so regularization is key
    n_features = X_feat_all.shape[1]
    
    alphas = np.logspace(-6, 2, 25)  # 1e-6 to 100
    
    for alpha in alphas:
        if time.time() > deadline:
            break
        try:
            # Solve (X^T X + alpha I) w = X^T y
            # Use Cholesky or standard solve
            A = X_tr.T @ X_tr + alpha * np.eye(n_features)
            b = X_tr.T @ y_tr
            w = solve(A, b, assume_a='pos')
            
            y_pred_val = X_val @ w
            mse = np.mean((y_pred_val - y_val) ** 2)
            
            if mse < best_val_mse:
                best_val_mse = mse
                best_type = 'ridge'
                best_model = {'alpha': alpha, 'w': w}
        except:
            continue

    # 2. Gaussian Process with Matern-5/2 kernel and Hyperparameter Optimization
    # Structural change: Instead of fixed RBF grid, optimize Matern kernel params
    
    def matern_52_kernel(X1, X2, length_scale, signal_var):
        # Distance matrix
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * (X1 @ X2.T)
        sq_dists = np.maximum(sq_dists, 0)
        dists = np.sqrt(sq_dists)
        
        # Matern 5/2 formula
        sqrt3_d = np.sqrt(3) * dists / length_scale
        K = signal_var * (1 + sqrt3_d + (sqrt3_d**2)/3.0) * np.exp(-sqrt3_d)
        
        # Add small jitter for numerical stability
        if X1.shape[0] == X2.shape[0] and X1.shape[1] == X2.shape[1]:
            # Only add jitter if it's the same set (training kernel)
            # Actually, we add jitter in the likelihood function to handle noise
            pass
        return K

    def neg_log_likelihood(params, X, y):
        length_scale = np.exp(params[0])
        signal_var = np.exp(params[1])
        noise_var = np.exp(params[2])
        
        K = matern_52_kernel(X, X, length_scale, signal_var)
        K += noise_var * np.eye(len(X))
        
        try:
            # Cholesky decomposition for stability and likelihood calculation
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L, y)
            alpha = np.linalg.solve(L.T, alpha)
            
            # Log likelihood: -0.5 * y^T K^-1 y - 0.5 * log(det(K)) - n/2 * log(2*pi)
            # log(det(K)) = 2 * sum(log(diag(L)))
            log_det = 2 * np.sum(np.log(np.diag(L)))
            ll = -0.5 * (y @ alpha + log_det) - len(y)/2 * np.log(2 * np.pi)
            
            return -ll
        except:
            return 1e6

    X_tr_norm = X_norm[train_idx]
    y_tr_gp = y_centered[train_idx]
    
    # Initial guesses for hyperparameters
    # length_scale: 0.1 to 1.0 (log space: -2 to 0)
    # signal_var: 1.0 (log space: 0)
    # noise_var: 0.01 (log space: -4)
    initial_params = np.array([0.0, 0.0, -4.0])
    
    try:
        if time.time() < deadline - 10: # Need time for optimization
            result = minimize(neg_log_likelihood, initial_params, args=(X_tr_norm, y_tr_gp), method='Nelder-Mead', options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4})
            
            if result.success and time.time() < deadline:
                # Evaluate validation performance with optimized params
                opt_ls = np.exp(result.x[0])
                opt_sv = np.exp(result.x[1])
                opt_nv = np.exp(result.x[2])
                
                # Compute prediction on validation set
                K_tr = matern_52_kernel(X_tr_norm, X_tr_norm, opt_ls, opt_sv)
                K_tr += opt_nv * np.eye(len(X_tr_norm))
                
                X_val_norm = X_norm[val_idx]
                K_val = matern_52_kernel(X_val_norm, X_tr_norm, opt_ls, opt_sv)
                
                try:
                    w = solve(K_tr, y_tr_gp, assume_a='pos')
                    y_pred_val = K_val @ w
                    mse = np.mean((y_pred_val - y_val) ** 2)
                    
                    if mse < best_val_mse:
                        best_val_mse = mse
                        best_type = 'gp_matern'
                        best_model = {'length_scale': opt_ls, 'signal_var': opt_sv, 'noise_var': opt_nv, 'w': w}
                except:
                    pass
    except:
        pass

    # Final Refit on ALL data with the best model type
    if best_type is None:
        # Default to ridge with a medium alpha
        best_type = 'ridge'
        best_model = {'alpha': 1.0, 'w': None}

    if best_type == 'ridge':
        alpha = best_model['alpha']
        # Refit on all training data
        A = X_feat_all.T @ X_feat_all + alpha * np.eye(n_features)
        b = X_feat_all.T @ y_centered
        try:
            w = solve(A, b, assume_a='pos')
        except:
            # Fallback to least squares if solve fails
            w = np.linalg.lstsq(X_feat_all, y_centered, rcond=None)[0]
            
        def predict(X):
            X_test_norm = (X - min_vals) / ranges
            X_test_feat_norm = generate_features(X_test_norm)
            X_test_feat_raw = generate_features_raw(X)
            X_test_feat_all = np.hstack([X_test_feat_norm, X_test_feat_raw])
            X_test_feat_all = np.nan_to_num(X_test_feat_all, nan=0.0, posinf=1e6, neginf=-1e6)
            return X_test_feat_all @ w + y_mean

    else: # GP Matern
        opt_ls = best_model['length_scale']
        opt_sv = best_model['signal_var']
        opt_nv = best_model['noise_var']
        
        K_train = matern_52_kernel(X_norm, X_norm, opt_ls, opt_sv)
        K_train += opt_nv * np.eye(len(X_norm))
        
        try:
            w = solve(K_train, y_centered, assume_a='pos')
        except:
            w = np.linalg.lstsq(K_train, y_centered, rcond=None)[0]

        def predict(X):
            X_test_norm = (X - min_vals) / ranges
            K_test = matern_52_kernel(X_test_norm, X_norm, opt_ls, opt_sv)
            return K_test @ w + y_mean

    return predict
