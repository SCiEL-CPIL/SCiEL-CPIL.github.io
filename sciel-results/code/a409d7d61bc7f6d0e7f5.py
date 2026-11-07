"""Symbolic regression with ensemble of Kernel Ridge Regressors and adaptive selection."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar

# Global time tracker
_total_start = None
_time_limit = 320.0  # Leave 10s safety margin before 330s kill

def _elapsed():
    global _total_start
    if _total_start is None:
        _total_start = time.time()
    return time.time() - _total_start

def _remaining():
    return _time_limit - _elapsed()

def _rbf_kernel(X1, X2, gamma):
    """Compute RBF kernel matrix between X1 and X2."""
    sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
    sq_dists = np.maximum(sq_dists, 0)
    return np.exp(-gamma * sq_dists)

def _poly_kernel(X1, X2, degree=3, gamma=1.0, coef0=1.0):
    """Compute polynomial kernel matrix between X1 and X2."""
    return (gamma * (X1 @ X2.T) + coef0) ** degree

def _linear_kernel(X1, X2):
    """Compute linear kernel matrix between X1 and X2."""
    return X1 @ X2.T

def _fit_krr(K, y, alpha):
    """Fit KRR given kernel matrix K and regularization alpha."""
    n = K.shape[0]
    A = K + alpha * np.eye(n)
    try:
        w = np.linalg.solve(A, y)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, y, rcond=None)[0]
    return w

def fit(X_train, y_train):
    global _total_start
    if _total_start is None:
        _total_start = time.time()
    
    n, d = X_train.shape
    X = X_train.astype(np.float64)
    y = y_train.astype(np.float64)
    
    # Normalize inputs to [0, 1] range for better kernel scaling
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-10] = 1.0
    X_norm = (X - X_min) / X_range
    
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    # Build ensemble of KRR models with different kernels and hyperparameters
    # We'll use cross-validation to select the best model or blend
    
    # Define kernel configurations
    kernel_configs = []
    
    # RBF kernels with different gammas
    for gamma in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        kernel_configs.append(('rbf', {'gamma': gamma}))
    
    # Polynomial kernels
    for degree in [2, 3, 4]:
        for gamma in [0.5, 1.0, 2.0]:
            kernel_configs.append(('poly', {'degree': degree, 'gamma': gamma, 'coef0': 1.0}))
    
    # Linear kernel
    kernel_configs.append(('linear', {}))
    
    # Compute all kernel matrices
    K_matrices = []
    for kernel_type, params in kernel_configs:
        if kernel_type == 'rbf':
            K = _rbf_kernel(X_norm, X_norm, params['gamma'])
        elif kernel_type == 'poly':
            K = _poly_kernel(X_norm, X_norm, **params)
        else:  # linear
            K = _linear_kernel(X_norm, X_norm)
        K_matrices.append(K)
    
    n_kernels = len(kernel_configs)
    
    # Hyperparameter search for alpha (regularization) using cross-validation
    alpha_values = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    
    # 5-fold cross-validation
    fold_size = n // 5
    folds = []
    for i in range(5):
        start = i * fold_size
        end = start + fold_size if i < 4 else n
        folds.append((start, end))
    
    best_cv_scores = np.full(n_kernels, np.inf)
    best_alphas = np.zeros(n_kernels)
    
    for k_idx in range(n_kernels):
        K = K_matrices[k_idx]
        
        for alpha in alpha_values:
            cv_scores = []
            for fold_idx, (start, end) in enumerate(folds):
                val_idx = np.arange(start, end)
                train_idx = np.concatenate([np.arange(0, start), np.arange(end, n)])
                
                K_train = K[np.ix_(train_idx, train_idx)]
                K_val = K[np.ix_(val_idx, train_idx)]
                y_train_fold = y_norm[train_idx]
                y_val = y_norm[val_idx]
                
                w = _fit_krr(K_train, y_train_fold, alpha)
                pred_val = K_val @ w
                mse_val = np.mean((pred_val - y_val) ** 2)
                cv_scores.append(mse_val)
            
            mean_cv = np.mean(cv_scores)
            if mean_cv < best_cv_scores[k_idx]:
                best_cv_scores[k_idx] = mean_cv
                best_alphas[k_idx] = alpha
        
        # Check time
        if _remaining() < 10.0:
            break
    
    # Select top models and create an ensemble
    # Sort by CV score
    sorted_indices = np.argsort(best_cv_scores)
    
    # Take top 3 models for ensemble
    n_ensemble = min(3, n_kernels)
    ensemble_indices = sorted_indices[:n_ensemble]
    
    # Fit final models on all data
    ensemble_weights = []
    ensemble_models = []
    
    for k_idx in ensemble_indices:
        K = K_matrices[k_idx]
        alpha = best_alphas[k_idx]
        w = _fit_krr(K, y_norm, alpha)
        ensemble_models.append((k_idx, w))
        
        # Weight inversely proportional to CV score
        weight = 1.0 / (best_cv_scores[k_idx] + 1e-10)
        ensemble_weights.append(weight)
    
    # Normalize weights
    total_weight = sum(ensemble_weights)
    if total_weight > 0:
        ensemble_weights = [w / total_weight for w in ensemble_weights]
    else:
        ensemble_weights = [1.0 / len(ensemble_weights)] * len(ensemble_weights)
    
    # Build prediction function
    def predict(X_test):
        X_t = X_test.astype(np.float64)
        X_t_norm = (X_t - X_min) / X_range
        
        y_pred = np.zeros(X_t.shape[0])
        for i, (k_idx, w) in enumerate(ensemble_models):
            kernel_type, params = kernel_configs[k_idx]
            if kernel_type == 'rbf':
                K_val = _rbf_kernel(X_t_norm, X_norm, params['gamma'])
            elif kernel_type == 'poly':
                K_val = _poly_kernel(X_t_norm, X_norm, **params)
            else:  # linear
                K_val = _linear_kernel(X_t_norm, X_norm)
            
            pred_i = K_val @ w
            y_pred += ensemble_weights[i] * pred_i
        
        return y_pred * y_std + y_mean
    
    return predict
# EVOLVE-BLOCK-END
