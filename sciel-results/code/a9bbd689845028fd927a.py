"""Symbolic regression via Random Fourier Features with L2 regularization."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    DEADLINE = 328.0  # Safety margin for 330s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    n, d = X.shape
    
    # Normalize inputs to [0, 1]
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1.0
    X_norm = (X - X_min) / X_range
    
    # Normalize target
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    # Random Fourier Features strategy
    # Sample multiple frequency scales and select best via cross-validation
    
    # Define frequency scales to try (log-spaced)
    # The true function's frequency content is unknown, so we try a range
    freq_scales = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0]
    
    # Number of random features per scale
    n_features = 512
    
    # Regularization strengths to try
    alphas = [1e-6, 1e-4, 1e-3, 1e-2, 0.1, 1.0]
    
    # We'll do a simple 5-fold CV to select the best (scale, alpha)
    # But to save time, let's first try a few scales and pick the best
    # via leave-out validation on a subset
    
    # Split for validation
    np.random.seed(42)
    perm = np.random.permutation(n)
    val_size = min(40, n // 5)
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]
    
    X_tr = X_norm[train_idx]
    y_tr = y_norm[train_idx]
    X_val = X_norm[val_idx]
    y_val = y_norm[val_idx]
    
    best_score = np.inf
    best_scale = 1.0
    best_alpha = 1e-3
    
    # Try each frequency scale
    for scale in freq_scales:
        # Generate random frequencies
        np.random.seed(int(scale * 1000))
        # For each dimension, sample frequencies from N(0, scale^2)
        # This corresponds to a Gaussian kernel with length scale ~ 1/scale
        W = np.random.randn(n_features, d) * scale
        b = np.random.randn(n_features) * 0.1
        
        # Build feature matrix for training
        # phi(x) = [cos(x @ W.T + b), sin(x @ W.T + b)]
        proj_tr = X_tr @ W.T + b  # (n_tr, n_features)
        phi_tr = np.concatenate([np.cos(proj_tr), np.sin(proj_tr)], axis=1)  # (n_tr, 2*n_features)
        
        # Build feature matrix for validation
        proj_val = X_val @ W.T + b  # (n_val, n_features)
        phi_val = np.concatenate([np.cos(proj_val), np.sin(proj_val)], axis=1)  # (n_val, 2*n_features)
        
        # Add bias term
        phi_tr = np.hstack([phi_tr, np.ones((len(phi_tr), 1))])
        phi_val = np.hstack([phi_val, np.ones((len(phi_val), 1))])
        
        # Try different regularization strengths
        for alpha in alphas:
            # Solve regularized least squares
            # (phi.T @ phi + alpha * I) w = phi.T @ y
            try:
                A = phi_tr.T @ phi_tr
                A += alpha * np.eye(A.shape[0])
                b_vec = phi_tr.T @ y_tr
                w = np.linalg.solve(A, b_vec)
                
                # Evaluate on validation
                pred_val = phi_val @ w
                mse_val = np.mean((pred_val - y_val) ** 2)
                
                if mse_val < best_score:
                    best_score = mse_val
                    best_scale = scale
                    best_alpha = alpha
            except np.linalg.LinAlgError:
                continue
        
        # Check time
        if time.time() > DEADLINE - 5:
            break
    
    # Now fit on all data with best parameters
    scale = best_scale
    alpha = best_alpha
    
    np.random.seed(int(scale * 1000))
    W = np.random.randn(n_features, d) * scale
    b = np.random.randn(n_features) * 0.1
    
    proj_all = X_norm @ W.T + b
    phi_all = np.concatenate([np.cos(proj_all), np.sin(proj_all)], axis=1)
    phi_all = np.hstack([phi_all, np.ones((len(phi_all), 1))])
    
    try:
        A = phi_all.T @ phi_all
        A += alpha * np.eye(A.shape[0])
        b_vec = phi_all.T @ y_norm
        w = np.linalg.solve(A, b_vec)
    except np.linalg.LinAlgError:
        # Fallback to simple linear model
        def predict_fallback(X_test):
            X_test = np.asarray(X_test, dtype=np.float64)
            X_test_norm = (X_test - X_min) / X_range
            # Simple linear + quadratic
            cols = [np.ones(len(X_test))]
            for j in range(d):
                cols.append(X_test_norm[:, j])
            for i in range(d):
                for j in range(i, d):
                    cols.append(X_test_norm[:, i] * X_test_norm[:, j])
            feat = np.column_stack(cols)
            w_ls, *_ = np.linalg.lstsq(np.column_stack([np.ones(n)] + [X_norm[:, j] for j in range(d)] + [X_norm[:, i] * X_norm[:, j] for i in range(d) for j in range(i, d)]), y, rcond=None)
            return feat @ w_ls
        
        return predict_fallback
    
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        X_test_norm = (X_test - X_min) / X_range
        
        proj_test = X_test_norm @ W.T + b
        phi_test = np.concatenate([np.cos(proj_test), np.sin(proj_test)], axis=1)
        phi_test = np.hstack([phi_test, np.ones((len(phi_test), 1))])
        
        mean_norm = phi_test @ w
        return mean_norm * y_std + y_mean
    
    return predict
# EVOLVE-BLOCK-END
