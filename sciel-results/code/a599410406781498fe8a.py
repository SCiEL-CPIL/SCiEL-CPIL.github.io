"""
Symbolic regression using rich basis expansion and cross-validated ridge regression.
"""
import numpy as np
import time
import warnings

warnings.filterwarnings("ignore")

# Global time management
_GLOBAL_START_TIME = None
TIME_LIMIT = 290.0  # Safety margin before 300s limit (330s kill, but be conservative)

def _get_time_left():
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
    return TIME_LIMIT - (time.time() - _GLOBAL_START_TIME)

def _generate_basis_features(x):
    """
    Generate a rich set of basis functions for 1D input x.
    Returns a matrix of shape (N, P) where P is the number of basis functions.
    """
    N = len(x)
    features = [np.ones(N)]  # Intercept
    
    # 1. Polynomials up to degree 10
    for k in range(1, 11):
        features.append(x ** k)
        
    # 2. Trigonometric functions with various frequencies
    freqs = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 1.0/np.pi, 2.0/np.pi]
    for f in freqs:
        features.append(np.sin(f * x))
        features.append(np.cos(f * x))
        
    # 3. Hyperbolic functions
    for scale in [0.5, 1.0, 2.0]:
        features.append(np.tanh(scale * x))
    
    # 4. Exponential functions
    for rate in [0.2, 0.5, 1.0, 2.0]:
        features.append(np.exp(-rate * x))
        features.append(np.exp(rate * x))
    features.append(np.exp(-x**2))       # Gaussian
    features.append(np.exp(-0.5 * x**2)) # Wider Gaussian
    
    # 5. Logarithmic (safe)
    features.append(np.log1p(np.abs(x)))
    features.append(np.sign(x) * np.log1p(np.abs(x)))
    
    # 6. Rational functions
    features.append(1.0 / (1.0 + x**2))
    features.append(1.0 / (1.0 + 0.25 * x**2))
    features.append(1.0 / (1.0 + 4.0 * x**2))
    
    # 7. Absolute value and related
    features.append(np.abs(x))
    features.append(x * np.abs(x))
    
    X_feat = np.column_stack(features)
    
    # Handle NaN/Inf
    X_feat = np.nan_to_num(X_feat, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X_feat

def _fit_ridge_svd(X, y, lam):
    """
    Fit Ridge Regression using SVD for numerical stability.
    min ||y - Xw||^2 + lam * ||w||^2
    Returns weights w.
    """
    N, P = X.shape
    try:
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        # Ridge solution: w = Vt^T * (s / (s^2 + lam)) * (U^T y)
        Ut_y = U.T @ y
        w = Vt.T @ (s / (s**2 + lam) * Ut_y)
        # Handle zero singular values
        mask = s > 1e-12
        if not np.all(mask):
            w = Vt.T @ np.where(mask, s / (s**2 + lam) * Ut_y, 0)
        return w
    except np.linalg.LinAlgError:
        # Fallback to lstsq
        A = X.T @ X + lam * np.eye(P)
        b = X.T @ y
        return np.linalg.lstsq(A, b, rcond=None)[0]

def _cross_validate_ridge(X_train, y_train, lams, n_folds=5):
    """
    Perform K-Fold Cross-Validation to find the best lambda.
    Returns best_lambda, best_mse, and a list of (lam, cv_mse) sorted by cv_mse.
    """
    n = X_train.shape[0]
    indices = np.arange(n)
    np.random.seed(42)
    indices = np.random.permutation(indices)
    folds = np.array_split(indices, n_folds)
    
    results = []
    
    for lam in lams:
        if _get_time_left() < 5.0:
            break
        fold_mses = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            
            X_tr = X_train[train_idx]
            y_tr = y_train[train_idx]
            X_val = X_train[val_idx]
            y_val = y_train[val_idx]
            
            # Fit
            w = _fit_ridge_svd(X_tr, y_tr, lam)
            pred = X_val @ w
            mse = np.mean((pred - y_val) ** 2)
            
            if not np.isfinite(mse):
                mse = np.inf
            fold_mses.append(mse)
            
        mean_mse = np.mean(fold_mses)
        results.append((lam, mean_mse))
    
    if not results:
        return lams[0], np.inf, []
    
    results.sort(key=lambda x: x[1])
    best_lam, best_mse = results[0]
    return best_lam, best_mse, results

def fit(X_train, y_train):
    global _GLOBAL_START_TIME
    if _GLOBAL_START_TIME is None:
        _GLOBAL_START_TIME = time.time()
        
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    n, d = X.shape
    
    # Ensure 1D handling for basis generation
    if d != 1:
        # Fallback for multi-dim: Simple RBF with fixed params
        gamma = 1.0 / (d * np.var(X, axis=0).mean() + 1e-8)
        lam = 1e-2
        K = np.exp(-gamma * np.sum((X[:, None, :] - X[None, :, :])**2, axis=2))
        K_reg = K + lam * np.eye(n)
        w = np.linalg.solve(K_reg, y)
        def predict(X_test):
            X_t = np.asarray(X_test, dtype=float)
            K_t = np.exp(-gamma * np.sum((X_t[:, None, :] - X[None, :, :])**2, axis=2))
            return K_t @ w
        return predict

    x = X[:, 0]
    
    # Normalize x for better numerical stability in basis generation
    x_mean = x.mean()
    x_std = x.std()
    if x_std < 1e-8:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std
    
    # Generate Basis
    X_basis = _generate_basis_features(x_norm)
    
    # Define Lambda grid - fewer points for speed
    lams = np.logspace(-3, 3, 30)
    
    # Check time
    if _get_time_left() < 10.0:
         # Fallback to simple linear
         X_simple = np.column_stack([np.ones(n), x])
         w = np.linalg.lstsq(X_simple, y, rcond=None)[0]
         def predict(X_test):
             X_t = np.asarray(X_test, dtype=float)
             x_t = X_t[:, 0]
             X_t_simple = np.column_stack([np.ones(len(x_t)), x_t])
             return X_t_simple @ w
         return predict

    # Cross-Validate to find best lambda
    best_lam, best_cv_mse, cv_results = _cross_validate_ridge(X_basis, y, lams)
    
    # Check time
    if _get_time_left() < 5.0:
         # Use best_lam from CV if it finished, else default
         if not cv_results:
             best_lam = 1.0

    # Fit final model with best lambda on ALL data
    w_final = _fit_ridge_svd(X_basis, y, best_lam)
    
    # Also fit a few other good lambdas for ensemble
    ensemble_weights = [(best_lam, w_final)]
    
    # Add top-3 CV performers for ensemble (if time permits)
    if _get_time_left() > 10.0 and len(cv_results) >= 2:
        for lam_i, mse_i in cv_results[1:4]:
            if _get_time_left() < 5.0:
                break
            w_i = _fit_ridge_svd(X_basis, y, lam_i)
            if np.all(np.isfinite(w_i)):
                ensemble_weights.append((lam_i, w_i))
    
    # Sanity check: if weights are huge or non-finite, fallback
    if not np.all(np.isfinite(w_final)):
         # Fallback to unregularized least squares with lower degree basis
         X_simple = np.column_stack([np.ones(n), x_norm, x_norm**2, x_norm**3])
         try:
             w_fallback = np.linalg.lstsq(X_simple, y, rcond=None)[0]
         except:
             w_fallback = np.zeros(X_simple.shape[1])
         
         def predict(X_test):
             X_t = np.asarray(X_test, dtype=float)
             x_t = X_t[:, 0]
             x_t_norm = (x_t - x_mean) / x_std
             X_t_simple = np.column_stack([np.ones(len(x_t)), x_t_norm, x_t_norm**2, x_t_norm**3])
             return X_t_simple @ w_fallback
         return predict

    def predict(X_test):
        X_t = np.asarray(X_test, dtype=float)
        x_t = X_t[:, 0]
        x_t_norm = (x_t - x_mean) / x_std
        X_t_basis = _generate_basis_features(x_t_norm)
        
        # Ensemble: average predictions from top models
        preds = []
        for lam_i, w_i in ensemble_weights:
            pred_i = X_t_basis @ w_i
            if np.all(np.isfinite(pred_i)):
                preds.append(pred_i)
        
        if not preds:
            return X_t_basis @ w_final
        
        return np.mean(preds, axis=0)

    return predict
