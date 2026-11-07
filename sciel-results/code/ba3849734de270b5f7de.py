"""Gaussian Process regression with Matérn-5/2 kernel and grid-search hyperparameters."""
# EVOLVE-BLOCK-START
import numpy as np


def fit(X_train, y_train):
    # X_train shape (n, d), y_train shape (n,)
    # All equations are 1D, so d=1
    n = X_train.shape[0]
    x = X_train[:, 0].copy()
    y = y_train.copy()
    
    # Center and scale x for numerical stability
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-10:
        x_std = 1.0
    x_scaled = (x - x_mean) / x_std
    
    # Center y for numerical stability
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std < 1e-10:
        y_std = 1.0
    y_scaled = (y - y_mean) / y_std
    
    def matern52_kernel(x1, x2, length_scale, signal_var):
        """Matérn 5/2 kernel for 1D inputs (already scaled)."""
        diff = x1[:, None] - x2[None, :]
        z = np.abs(diff) / length_scale
        # Matérn 5/2: (1 + sqrt(5)*z + 5/3*z^2) * exp(-sqrt(5)*z)
        kernel = (1.0 + np.sqrt(5.0) * z + (5.0 / 3.0) * z**2) * np.exp(-np.sqrt(5.0) * z)
        return signal_var * kernel
    
    # Grid search for hyperparameters
    # Length scale in scaled space
    ls_grid = np.logspace(-1.0, 2.0, 20)
    # Signal variance in scaled space
    sv_grid = np.logspace(-2.0, 2.0, 10)
    
    best_mse = np.inf
    best_ls = 1.0
    best_sv = 1.0
    
    # Simple 5-fold cross-validation
    indices = np.arange(n)
    np.random.seed(42) # For reproducibility in CV split if needed, though deterministic split is better
    # Deterministic 5-fold split
    fold_size = n // 5
    folds = [indices[i*fold_size:(i+1)*fold_size] for i in range(5)]
    if n % 5 != 0:
        # Append remainder to the last fold
        folds[-1] = np.concatenate([folds[-1], indices[5*fold_size:]])
        
    for ls in ls_grid:
        for sv in sv_grid:
            cv_errors = []
            for i in range(5):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
                
                x_tr = x_scaled[train_idx]
                y_tr = y_scaled[train_idx]
                x_va = x_scaled[val_idx]
                y_va = y_scaled[val_idx]
                
                # Noise variance is small for noiseless data
                nv = 1e-5
                
                K_train = matern52_kernel(x_tr, x_tr, ls, sv) + nv * np.eye(len(x_tr))
                K_train += 1e-6 * np.eye(len(x_tr)) # Jitter
                
                try:
                    L = np.linalg.cholesky(K_train)
                except np.linalg.LinAlgError:
                    continue
                    
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr))
                
                K_star = matern52_kernel(x_va, x_tr, ls, sv)
                y_pred = K_star @ alpha
                
                mse = np.mean((y_pred - y_va)**2)
                cv_errors.append(mse)
            
            mean_mse = np.mean(cv_errors)
            if mean_mse < best_mse:
                best_mse = mean_mse
                best_ls = ls
                best_sv = sv
                
    # Final GP fit using all training data with best hyperparameters
    best_nv = 1e-5
    
    K = matern52_kernel(x_scaled, x_scaled, best_ls, best_sv) + best_nv * np.eye(n)
    
    # Add jitter for numerical stability
    K = K + 1e-6 * np.eye(n)
    
    try:
        L = np.linalg.cholesky(K)
    except np.linalg.LinAlgError:
        K = K + 1e-4 * np.eye(n)
        L = np.linalg.cholesky(K)
    
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_scaled))
    
    # Create predict function
    def predict(X):
        x_new = X[:, 0].copy()
        x_new_scaled = (x_new - x_mean) / x_std
        
        K_star = matern52_kernel(x_new_scaled, x_scaled, best_ls, best_sv)
        y_pred_scaled = K_star @ alpha
        y_pred = y_pred_scaled * y_std + y_mean
        return y_pred
    
    return predict
# EVOLVE-BLOCK-END
