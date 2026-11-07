"""
Strategy: Kernel Regression with Adaptive Bandwidth Selection.
Instead of piecewise splines or fixed basis functions, we use a non-parametric
approach: Nadaraya-Watson kernel regression. We search over bandwidth parameters
using cross-validation to find the optimal smoothness level. This approach can
adapt to local features of the function while maintaining global smoothness,
and is particularly effective for 1D regression with moderate sample sizes.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import interp1d

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280  # Leave 20s buffer for 300s limit
    
    X = X_train[:, 0]  # 1D input
    y = y_train
    
    n = len(X)
    
    # Sort data for efficient kernel regression
    sort_idx = np.argsort(X)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    
    # Search over bandwidth parameters
    # Bandwidth controls the effective number of neighbors used
    # Too small: high variance (overfitting)
    # Too large: high bias (underfitting)
    # We'll search over a range of bandwidths
    bandwidths = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    
    best_score = np.inf
    best_bw = 1.0
    best_predict = None
    
    # Cross-validation to select best bandwidth
    n_folds = 5
    fold_size = n // n_folds
    
    for bw in bandwidths:
        if time.time() > deadline:
            break
        
        try:
            cv_errors = []
            
            for i in range(n_folds):
                val_start = i * fold_size
                val_end = (i + 1) * fold_size if i < n_folds - 1 else n
                val_idx = np.arange(val_start, val_end)
                train_idx = np.concatenate([np.arange(0, val_start), np.arange(val_end, n)])
                
                X_val = X[val_idx]
                y_val = y[val_idx]
                X_tr = X[train_idx]
                y_tr = y[train_idx]
                
                # Sort training data
                sort_idx_tr = np.argsort(X_tr)
                X_tr_sorted = X_tr[sort_idx_tr]
                y_tr_sorted = y_tr[sort_idx_tr]
                
                # Predict on validation data using kernel regression
                pred = np.zeros(len(X_val))
                
                for j in range(len(X_val)):
                    x_val = X_val[j]
                    # Compute Gaussian kernel weights
                    dist = np.abs(X_tr_sorted - x_val)
                    weights = np.exp(-0.5 * (dist / bw) ** 2)
                    
                    # Check if weights are too small (numerical issues)
                    if np.sum(weights) < 1e-10:
                        # Fall back to nearest neighbor
                        nearest_idx = np.argmin(dist)
                        pred[j] = y_tr_sorted[nearest_idx]
                    else:
                        # Nadaraya-Watson estimator
                        pred[j] = np.sum(weights * y_tr_sorted) / np.sum(weights)
                
                if not np.all(np.isfinite(pred)):
                    cv_errors.append(np.inf)
                else:
                    mse = np.mean((pred - y_val) ** 2)
                    cv_errors.append(mse)
            
            mean_mse = np.mean(cv_errors)
            
            if mean_mse < best_score:
                best_score = mean_mse
                best_bw = bw
                
        except Exception:
            continue
    
    # Create the final predict function using the best bandwidth
    def predict(X, X_sorted=X_sorted, y_sorted=y_sorted, bw=best_bw, mean_y=np.mean(y)):
        X_flat = X.flatten()
        pred = np.zeros(len(X_flat))
        
        for j in range(len(X_flat)):
            x_val = X_flat[j]
            # Compute Gaussian kernel weights
            dist = np.abs(X_sorted - x_val)
            weights = np.exp(-0.5 * (dist / bw) ** 2)
            
            # Check if weights are too small
            if np.sum(weights) < 1e-10:
                # Fall back to nearest neighbor
                nearest_idx = np.argmin(dist)
                pred[j] = y_sorted[nearest_idx]
            else:
                # Nadaraya-Watson estimator
                pred[j] = np.sum(weights * y_sorted) / np.sum(weights)
        
        # Ensure all predictions are finite
        pred = np.where(np.isfinite(pred), pred, mean_y)
        
        return pred.reshape(X.shape)
    
    return predict
# EVOLVE-BLOCK-END
