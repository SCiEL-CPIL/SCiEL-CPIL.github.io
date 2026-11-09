"""
Symbolic regression using Kernel Ridge Regression with RBF kernel.
Uses hyperparameter tuning via hold-out validation on the training set.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    t_start = time.time()
    deadline = 280.0  # Safety margin before 300s limit

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape

    # If n is very small, handle edge case, but n=200 is guaranteed
    if n < 2:
        mean_y = np.mean(y) if n > 0 else 0.0
        return lambda X_test: np.full(len(X_test), mean_y)

    # Define RBF kernel function
    def rbf_kernel(X1, X2, l):
        # X1: (n1, d), X2: (n2, d)
        # Compute squared distances
        X1_sq = np.sum(X1 ** 2, axis=1)[:, np.newaxis]
        X2_sq = np.sum(X2 ** 2, axis=1)[np.newaxis, :]
        dist_sq = X1_sq + X2_sq - 2 * X1 @ X2.T
        # Clip to avoid negative values due to floating point errors
        dist_sq = np.maximum(dist_sq, 0)
        return np.exp(-dist_sq / (2 * l ** 2))

    # Pre-compute some statistics for hyperparameter search
    y_mean = np.mean(y)
    y_centered = y - y_mean
    
    # Standardize X for better kernel conditioning? 
    # RBF is scale-sensitive. Let's standardize X.
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    # Avoid division by zero
    X_std[X_std < 1e-10] = 1.0
    X_norm = (X - X_mean) / X_std

    # Hyperparameter search
    # Length scales: from small (high frequency) to large (low frequency)
    # Since data is noiseless, we want to fit closely, so smaller l is often better,
    # but too small causes overfitting if test points are not exactly on the manifold.
    # However, test points are from the same distribution.
    # Let's search a range of l.
    # Note: X is standardized, so l is in standardized units.
    l_candidates = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    alpha_candidates = [1e-8, 1e-6, 1e-4, 1e-2, 1.0, 10.0]
    
    best_val_mse = np.inf
    best_l = 1.0
    best_alpha = 1e-6
    
    # Simple hold-out validation: 80% train, 20% val
    # Shuffle indices
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    
    val_size = max(10, int(0.2 * n))
    train_size = n - val_size
    
    X_val = X_norm[indices[:val_size]]
    y_val = y[indices[:val_size]]
    X_tr = X_norm[indices[val_size:]]
    y_tr = y[indices[val_size:]]
    
    # Pre-compute kernel for training set for each l? 
    # K_tr is (n_tr, n_tr)
    # We can solve (K + alpha*I) w = y_tr
    # Then predict on val: K_val_tr @ w
    # MSE = mean((pred - y_val)^2)
    
    n_tr = len(y_tr)
    
    for l in l_candidates:
        if time.time() > deadline:
            break
        K_tr = rbf_kernel(X_tr, X_tr, l)
        # Add ridge penalty
        for alpha in alpha_candidates:
            if time.time() > deadline:
                break
            try:
                # Solve linear system
                # K_tr is symmetric positive definite
                A = K_tr + alpha * np.eye(n_tr)
                w = np.linalg.solve(A, y_tr)
                
                # Predict on validation set
                K_val_tr = rbf_kernel(X_val, X_tr, l)
                y_pred_val = K_val_tr @ w
                
                val_mse = np.mean((y_pred_val - y_val) ** 2)
                
                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    best_l = l
                    best_alpha = alpha
                    
            except np.linalg.LinAlgError:
                continue

    # Refine search around best found? 
    # For now, use the best found. 
    # If best_val_mse is still inf, fallback to defaults
    if np.isinf(best_val_mse):
        best_l = 1.0
        best_alpha = 1e-6

    # Final fit on ALL training data with best hyperparameters
    K_full = rbf_kernel(X_norm, X_norm, best_l)
    A = K_full + best_alpha * np.eye(n)
    try:
        w = np.linalg.solve(A, y)
    except np.linalg.LinAlgError:
        # Fallback to pseudo-inverse if singular
        w = np.linalg.pinv(A) @ y

    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        
        # Normalize test data using training statistics
        X_test_norm = (X_test - X_mean) / X_std
        
        # Compute kernel between test and train
        K_test_train = rbf_kernel(X_test_norm, X_norm, best_l)
        
        # Predict
        y_pred = K_test_train @ w
        
        # Ensure finite
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return y_pred

    return predict
# EVOLVE-BLOCK-END
