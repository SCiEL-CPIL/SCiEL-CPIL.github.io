"""Gaussian Process regression with Matérn-5/2 kernel and grid-search hyperparameter tuning."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve, cho_inv


def fit(X_train, y_train):
    # Flatten X to 1D since all equations are univariate
    x_train = X_train.flatten()
    y_train = y_train.flatten()
    
    # Sort by x for better conditioning (optional but can help)
    sort_idx = np.argsort(x_train)
    x_sorted = x_train[sort_idx]
    y_sorted = y_train[sort_idx]
    
    # Grid search over length-scale and signal variance
    # Length-scale: geometric spacing
    length_scales = np.logspace(-1.5, 1.5, 15)
    # Signal variance: geometric spacing
    signal_variances = np.logspace(-2, 2, 10)
    
    # Noise variance is small but non-zero for numerical stability
    noise_var = 1e-6
    
    best_mse = np.inf
    best_l = None
    best_sigma_f = None
    
    # 5-fold cross-validation
    n = len(x_sorted)
    n_folds = 5
    fold_size = n // n_folds
    indices = np.arange(n)
    
    for l in length_scales:
        for sigma_f in signal_variances:
            total_mse = 0.0
            valid = True
            
            for fold in range(n_folds):
                val_start = fold * fold_size
                val_end = val_start + fold_size if fold < n_folds - 1 else n
                val_idx = indices[val_start:val_end]
                train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
                
                x_val = x_sorted[val_idx]
                y_val = y_sorted[val_idx]
                x_tr = x_sorted[train_idx]
                y_tr = y_sorted[train_idx]
                
                # Compute kernel matrix
                K = matern52_kernel(x_tr, x_tr, l, sigma_f)
                K += noise_var * np.eye(len(x_tr))
                
                try:
                    # Cholesky decomposition
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    valid = False
                    break
                
                # Solve for alpha
                alpha = cho_solve((L, True), y_tr)
                
                # Compute kernel between train and val
                K_s = matern52_kernel(x_val, x_tr, l, sigma_f)
                
                # Predict
                y_pred = K_s @ alpha
                
                mse = np.mean((y_pred - y_val) ** 2)
                total_mse += mse
            
            if valid:
                mean_mse = total_mse / n_folds
                if mean_mse < best_mse:
                    best_mse = mean_mse
                    best_l = l
                    best_sigma_f = sigma_f
    
    # If no valid combination found, use defaults
    if best_l is None:
        best_l = 1.0
        best_sigma_f = 1.0
    
    # Final fit with best hyperparameters on all data
    K = matern52_kernel(x_sorted, x_sorted, best_l, best_sigma_f)
    K += noise_var * np.eye(n)
    
    try:
        L = np.linalg.cholesky(K)
        alpha = cho_solve((L, True), y_sorted)
    except np.linalg.LinAlgError:
        # Fallback to least squares polynomial if GP fails
        def features(X):
            x = X.flatten()
            cols = [np.ones(len(x))] + [x ** i for i in range(1, 6)]
            return np.column_stack(cols)
        w, *_ = np.linalg.lstsq(features(X_train), y_train, rcond=None)
        return lambda X: features(X) @ w
    
    def predict(X):
        x = X.flatten()
        K_s = matern52_kernel(x, x_sorted, best_l, best_sigma_f)
        return K_s @ alpha
    
    return predict


def matern52_kernel(x1, x2, l, sigma_f):
    """Matérn 5/2 kernel for 1D inputs."""
    # x1: (n,), x2: (m,)
    # Compute pairwise distances
    diff = x1[:, np.newaxis] - x2[np.newaxis, :]
    r = np.abs(diff)
    sqrt3_r = np.sqrt(3.0) * r / l
    K = sigma_f ** 2 * (1.0 + sqrt3_r + sqrt3_r ** 2 / 3.0) * np.exp(-sqrt3_r)
    return K
