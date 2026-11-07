"""
Symbolic regression using adaptive kernel portfolio + basis expansion with ensemble selection.
"""
import numpy as np
import time
import warnings
from scipy.linalg import solve

warnings.filterwarnings("ignore")

_GLOBAL_START_TIME = None
TIME_LIMIT = 280.0  # Safety margin before 330s kill

def _get_time_left():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    return TIME_LIMIT - (time.time() - _GLOBAL_START_TIME)

def _generate_basis_features(x):
    """
    Generate a rich set of basis functions for 1D input x.
    Returns a matrix of shape (N, P).
    """
    N = len(x)
    features = [np.ones(N)]
    
    # Polynomials up to degree 12
    for k in range(1, 13):
        features.append(x ** k)
        
    # Trigonometric functions
    freqs = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    for f in freqs:
        features.append(np.sin(f * x))
        features.append(np.cos(f * x))
        
    # Hyperbolic functions
    for scale in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]:
        features.append(np.tanh(scale * x))
    
    # Exponential functions
    for rate in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0]:
        features.append(np.exp(-rate * x))
        features.append(np.exp(rate * x))
    features.append(np.exp(-x**2))
    features.append(np.exp(-0.5 * x**2))
    features.append(np.exp(-2.0 * x**2))
    
    # Logarithmic
    features.append(np.log1p(np.abs(x)))
    features.append(np.sign(x) * np.log1p(np.abs(x)))
    
    # Rational functions
    for c in [0.5, 1.0, 2.0, 4.0]:
        features.append(1.0 / (1.0 + c * x**2))
    
    # Absolute value and related
    features.append(np.abs(x))
    features.append(x * np.abs(x))
    features.append(x * np.tanh(x))
    
    X_feat = np.column_stack(features)
    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X_feat

def _compute_rbf_kernel(X1, X2, gamma):
    """Compute RBF kernel matrix between X1 and X2."""
    # Efficient computation of squared distances
    X1_sq = np.sum(X1**2, axis=1)[:, None]
    X2_sq = np.sum(X2**2, axis=1)[None, :]
    dist_sq = X1_sq + X2_sq - 2 * X1 @ X2.T
    dist_sq = np.maximum(dist_sq, 0)
    return np.exp(-gamma * dist_sq)

def _fit_ridge_svd(X, y, lam):
    """Fit Ridge Regression using SVD for numerical stability."""
    N, P = X.shape
    try:
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        Ut_y = U.T @ y
        mask = s > 1e-12
        w = Vt.T @ np.where(mask, s / (s**2 + lam) * Ut_y, 0)
        return w
    except np.linalg.LinAlgError:
        A = X.T @ X + lam * np.eye(P)
        b = X.T @ y
        return np.linalg.lstsq(A, b, rcond=None)[0]

def _fit_kernel_ridge(X_train, y_train, kernel_type, kernel_params, lam):
    """
    Fit kernel ridge regression.
    kernel_type: 'rbf', 'poly', 'sine'
    Returns (predict_fn, w) where w are the dual coefficients.
    """
    n = X_train.shape[0]
    
    if kernel_type == 'rbf':
        gamma = kernel_params.get('gamma', 1.0)
        K = _compute_rbf_kernel(X_train, X_train, gamma)
    elif kernel_type == 'poly':
        degree = kernel_params.get('degree', 3)
        coef0 = kernel_params.get('coef0', 1.0)
        K = (X_train @ X_train.T + coef0) ** degree
    elif kernel_type == 'sine':
        p = kernel_params.get('p', 1.0)
        q = kernel_params.get('q', 1.0)
        K = np.sin(p * (X_train @ X_train.T + q))
    else:
        K = np.ones((n, n))
    
    K_reg = K + lam * np.eye(n)
    
    try:
        w = solve(K_reg, y, assume_a='pos')
    except:
        w = np.linalg.lstsq(K_reg, y, rcond=None)[0]
    
    if not np.all(np.isfinite(w)):
        w = np.zeros(n)
    
    def predict(X_test, _X_train=X_train, _w=w, _kernel_type=kernel_type, 
                _kernel_params=kernel_params, _lam=lam):
        if _kernel_type == 'rbf':
            gamma = _kernel_params.get('gamma', 1.0)
            K_t = _compute_rbf_kernel(X_test, _X_train, gamma)
        elif _kernel_type == 'poly':
            degree = _kernel_params.get('degree', 3)
            coef0 = _kernel_params.get('coef0', 1.0)
            K_t = (X_test @ _X_train.T + coef0) ** degree
        elif _kernel_type == 'sine':
            p = _kernel_params.get('p', 1.0)
            q = _kernel_params.get('q', 1.0)
            K_t = np.sin(p * (X_test @ _X_train.T + q))
        else:
            K_t = np.ones((X_test.shape[0], n))
        
        pred = K_t @ _w
        return np.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
    
    return predict, w

def _cross_validate_kernel(X_train, y_train, kernel_configs, n_folds=5):
    """
    Cross-validate multiple kernel configurations.
    Returns list of (config, cv_mse, predict_fn) sorted by cv_mse.
    """
    n = X_train.shape[0]
    indices = np.arange(n)
    np.random.seed(42)
    indices = np.random.permutation(indices)
    folds = np.array_split(indices, n_folds)
    
    results = []
    
    for config in kernel_configs:
        if _get_time_left() < 10.0:
            break
        
        kernel_type = config['type']
        kernel_params = config.get('params', {})
        lam = config.get('lam', 1e-2)
        
        fold_mses = []
        for i in range(n_folds):
            if _get_time_left() < 5.0:
                break
            
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            
            X_tr = X_train[train_idx]
            y_tr = y_train[train_idx]
            X_val = X_train[val_idx]
            y_val = y_train[val_idx]
            
            pred_fn, _ = _fit_kernel_ridge(X_tr, y_tr, kernel_type, kernel_params, lam)
            pred = pred_fn(X_val)
            mse = np.mean((pred - y_val) ** 2)
            
            if not np.isfinite(mse):
                mse = np.inf
            fold_mses.append(mse)
        
        if len(fold_mses) == n_folds:
            mean_mse = np.mean(fold_mses)
            # Refit on all data
            pred_fn, _ = _fit_kernel_ridge(X_train, y_train, kernel_type, kernel_params, lam)
            results.append((config, mean_mse, pred_fn))
    
    results.sort(key=lambda x: x[1])
    return results

def _cross_validate_basis(X_train, y_train, lams, n_folds=5):
    """
    Cross-validate basis expansion with different regularization strengths.
    Returns list of (lam, cv_mse, weights) sorted by cv_mse.
    """
    n = X_train.shape[0]
    indices = np.arange(n)
    np.random.seed(42)
    indices = np.random.permutation(indices)
    folds = np.array_split(indices, n_folds)
    
    # Precompute basis
    X_basis = _generate_basis_features(X_train)
    
    results = []
    
    for lam in lams:
        if _get_time_left() < 5.0:
            break
        
        fold_mses = []
        for i in range(n_folds):
            if _get_time_left() < 3.0:
                break
            
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            
            X_tr = X_basis[train_idx]
            y_tr = y_train[train_idx]
            X_val = X_basis[val_idx]
            y_val = y_train[val_idx]
            
            w = _fit_ridge_svd(X_tr, y_tr, lam)
            pred = X_val @ w
            mse = np.mean((pred - y_val) ** 2)
            
            if not np.isfinite(mse):
                mse = np.inf
            fold_mses.append(mse)
        
        if len(fold_mses) == n_folds:
            mean_mse = np.mean(fold_mses)
            w_final = _fit_ridge_svd(X_basis, y_train, lam)
            results.append((lam, mean_mse, w_final))
    
    results.sort(key=lambda x: x[1])
    return results

def fit(X_train, y_train):
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    n, d = X.shape
    
    # Ensure 1D handling
    if d != 1:
        # Fallback for multi-dim: Simple RBF
        x_range = X.max(axis=0) - X.min(axis=0)
        x_range = np.maximum(x_range, 1e-8)
        gamma = 1.0 / (d * np.mean(x_range**2) + 1e-8)
        lam = 1e-2
        K = _compute_rbf_kernel(X, X, gamma)
        K_reg = K + lam * np.eye(n)
        w = solve(K_reg, y, assume_a='pos')
        
        def predict(X_test):
            X_t = np.asarray(X_test, dtype=float)
            K_t = _compute_rbf_kernel(X_t, X, gamma)
            return np.nan_to_num(K_t @ w, nan=0.0, posinf=0.0, neginf=0.0)
        return predict

    x = X[:, 0]
    
    # Normalize x for basis generation
    x_mean = x.mean()
    x_std = x.std()
    if x_std < 1e-8:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std
    
    # Strategy 1: Kernel Ridge Regression portfolio
    kernel_configs = []
    
    # RBF kernels with various gammas
    x_range = x_norm.max() - x_norm.min()
    base_gamma = 1.0 / (x_range**2 + 1e-8)
    for g_mult in [0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]:
        for lam in [1e-4, 1e-3, 1e-2, 1e-1, 1.0]:
            kernel_configs.append({
                'type': 'rbf',
                'params': {'gamma': g_mult * base_gamma},
                'lam': lam
            })
    
    # Polynomial kernels
    for degree in [2, 3, 4]:
        for coef0 in [0.5, 1.0, 2.0]:
            for lam in [1e-3, 1e-2, 1e-1]:
                kernel_configs.append({
                    'type': 'poly',
                    'params': {'degree': degree, 'coef0': coef0},
                    'lam': lam
                })
    
    # Sine kernels
    for p in [0.5, 1.0, 2.0]:
        for q in [0.5, 1.0, 2.0]:
            for lam in [1e-3, 1e-2, 1e-1]:
                kernel_configs.append({
                    'type': 'sine',
                    'params': {'p': p, 'q': q},
                    'lam': lam
                })
    
    # Strategy 2: Basis expansion with ridge regression
    lams = np.logspace(-4, 2, 25)
    
    # Time management: run kernel CV first, then basis CV
    kernel_results = []
    basis_results = []
    
    if _get_time_left() > 30.0:
        kernel_results = _cross_validate_kernel(X, y, kernel_configs[:40])
    
    if _get_time_left() > 15.0:
        basis_results = _cross_validate_basis(X, y, lams)
    
    # Build ensemble from top performers
    ensemble_predictors = []
    
    # Add top kernel models
    for config, cv_mse, pred_fn in kernel_results[:5]:
        if cv_mse < np.inf:
            ensemble_predictors.append(pred_fn)
    
    # Add top basis models
    for lam, cv_mse, w in basis_results[:3]:
        if cv_mse < np.inf:
            def basis_pred(X_test, _w=w, _x_mean=x_mean, _x_std=x_std):
                X_t = np.asarray(X_test, dtype=float)
                x_t = X_t[:, 0]
                x_t_norm = (x_t - _x_mean) / _x_std
                X_t_basis = _generate_basis_features(x_t_norm)
                return np.nan_to_num(X_t_basis @ _w, nan=0.0, posinf=0.0, neginf=0.0)
            ensemble_predictors.append(basis_pred)
    
    # If no good models, use a simple fallback
    if not ensemble_predictors:
        # Simple polynomial fit
        X_simple = np.column_stack([np.ones(n), x_norm, x_norm**2, x_norm**3, x_norm**4])
        w_fallback = np.linalg.lstsq(X_simple, y, rcond=None)[0]
        
        def predict(X_test):
            X_t = np.asarray(X_test, dtype=float)
            x_t = X_t[:, 0]
            x_t_norm = (x_t - x_mean) / x_std
            X_t_simple = np.column_stack([np.ones(len(x_t)), x_t_norm, x_t_norm**2, x_t_norm**3, x_t_norm**4])
            return np.nan_to_num(X_t_simple @ w_fallback, nan=0.0, posinf=0.0, neginf=0.0)
        return predict
    
    # Weighted ensemble: weight inversely proportional to CV MSE
    all_cv_mses = []
    for config, cv_mse, pred_fn in kernel_results[:5]:
        all_cv_mses.append(cv_mse)
    for lam, cv_mse, w in basis_results[:3]:
        all_cv_mses.append(cv_mse)
    
    # Convert to weights (inverse MSE, with floor to avoid division by zero)
    all_cv_mses = np.array(all_cv_mses)
    all_cv_mses = np.maximum(all_cv_mses, 1e-10)
    weights = 1.0 / all_cv_mses
    weights = weights / np.sum(weights)
    
    def predict(X_test):
        X_t = np.asarray(X_test, dtype=float)
        
        preds = []
        for pred_fn in ensemble_predictors:
            pred = pred_fn(X_t)
            if np.all(np.isfinite(pred)):
                preds.append(pred)
        
        if not preds:
            # Ultimate fallback
            return np.zeros(X_t.shape[0])
        
        # Weighted average
        n_models = len(ensemble_predictors)
        # Use only the weights corresponding to models that produced finite predictions
        # For simplicity, use equal weighting if we can't easily track which models succeeded
        if len(preds) == n_models:
            weighted_pred = sum(w * p for w, p in zip(weights, preds))
        else:
            # If some models failed, use equal weighting on successful ones
            weighted_pred = np.mean(preds, axis=0)
        
        return np.nan_to_num(weighted_pred, nan=0.0, posinf=0.0, neginf=0.0)
    
    return predict
