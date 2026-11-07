"""Symbolic regression via Matérn-5/2 GP with CV-based hyperparameter search."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Global deadline: 330s wall-clock, use 310s as safety margin
_DEADLINE = time.time() + 310.0


def _remaining():
    return _DEADLINE - time.time()


def _matern52_kernel(x1, x2, length_scale, variance):
    """Matérn-5/2 kernel for 1D inputs."""
    d = np.abs(x1[:, None] - x2[None, :]) / length_scale
    sqrt3 = np.sqrt(3.0)
    return variance * (1.0 + sqrt3 * d + (d ** 2) / 3.0) * np.exp(-sqrt3 * d)


def _fit_gp_matern(x, y, length_scale, variance, noise=1e-10):
    """Fit GP with Matérn-5/2 kernel, return predict function or None on failure."""
    n = len(x)
    K = _matern52_kernel(x, x, length_scale, variance) + noise * np.eye(n)
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    except np.linalg.LinAlgError:
        return None

    def predict(x_new):
        K_star = _matern52_kernel(x_new, x, length_scale, variance)
        return K_star @ alpha
    return predict


def _cross_validate_gp(x, y, length_scale, variance, n_folds=5, noise=1e-10):
    """5-fold CV for GP with Matérn-5/2 kernel. Returns mean CV MSE or inf on failure."""
    n = len(x)
    if n < n_folds * 2:
        return np.inf
    
    # Stratified-ish split: sort by x, then assign to folds in round-robin
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    
    fold_ids = np.arange(n) % n_folds
    
    fold_mses = []
    for fold in range(n_folds):
        if _remaining() < 2:
            return np.inf
        
        test_mask = (fold_ids == fold)
        train_mask = ~test_mask
        
        x_tr = x_sorted[train_mask]
        y_tr = y_sorted[train_mask]
        x_te = x_sorted[test_mask]
        y_te = y_sorted[test_mask]
        
        pred_fn = _fit_gp_matern(x_tr, y_tr, length_scale, variance, noise)
        if pred_fn is None:
            return np.inf
        
        pred_te = pred_fn(x_te)
        mse = np.mean((pred_te - y_te) ** 2)
        fold_mses.append(mse)
    
    return np.mean(fold_mses)


def _poly_fit(X, y, degree):
    """Polynomial fit of given degree."""
    n = len(X)
    x = X[:, 0]
    if degree == 0:
        c = np.mean(y)
        return lambda X_new: np.full(len(X_new), c)
    cols = [x ** k for k in range(degree + 1)]
    F = np.column_stack(cols)
    w, *_ = np.linalg.lstsq(F, y, rcond=None)
    def predict(X_new):
        x_new = X_new[:, 0]
        F_new = np.column_stack([x_new ** k for k in range(degree + 1)])
        return F_new @ w
    return predict


def fit(X_train, y_train):
    """Fit GP with Matérn-5/2 kernel using CV-based hyperparameter search."""
    X = X_train.astype(np.float64)
    y = y_train.astype(np.float64)
    x = X[:, 0]
    n = len(x)
    
    # Normalize x to [0, 1] for stable kernel evaluation
    x_min, x_max = x.min(), x.max()
    x_range = x_max - x_min + 1e-10
    x_norm = (x - x_min) / x_range
    
    # Standardize y
    y_mean = np.mean(y)
    y_std = np.std(y) + 1e-10
    y_norm = (y - y_mean) / y_std
    
    candidates = []  # (cv_mse, predict_fn)
    
    # Grid search over length_scale (in normalized space) and signal variance
    # Length scales in normalized [0,1] space: small = wiggly, large = smooth
    ls_grid = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.8, 1.5]
    var_grid = [0.5, 1.0, 2.0, 5.0]
    
    best_cv_mse = np.inf
    best_ls = 0.1
    best_var = 1.0
    
    for ls in ls_grid:
        if _remaining() < 10:
            break
        for var in var_grid:
            if _remaining() < 5:
                break
            try:
                cv_mse = _cross_validate_gp(x_norm, y_norm, ls, var, n_folds=5)
                if cv_mse < best_cv_mse and np.isfinite(cv_mse):
                    best_cv_mse = cv_mse
                    best_ls = ls
                    best_var = var
            except Exception:
                pass
    
    # Fit final GP on all data with best hyperparameters
    pred_fn_norm = _fit_gp_matern(x_norm, y_norm, best_ls, best_var, noise=1e-10)
    
    if pred_fn_norm is not None:
        def predict(X_new):
            x_new = X_new[:, 0].astype(np.float64)
            x_new_norm = (x_new - x_min) / x_range
            try:
                pred_norm = pred_fn_norm(x_new_norm)
                pred = pred_norm * y_std + y_mean
                pred = np.asarray(pred, dtype=np.float64).flatten()
                if not np.all(np.isfinite(pred)):
                    pred = np.nan_to_num(pred, nan=y_mean, posinf=1e6, neginf=-1e6)
                return pred
            except Exception:
                return np.full(len(X_new), y_mean)
    else:
        # Fallback: polynomial fit
        pred_fn = _poly_fit(X, y, 4)
        def predict(X_new):
            try:
                result = np.asarray(pred_fn(X_new), dtype=np.float64).flatten()
                if not np.all(np.isfinite(result)):
                    result = np.nan_to_num(result, nan=np.mean(y), posinf=1e6, neginf=-1e6)
                return result
            except Exception:
                return np.full(len(X_new), np.mean(y))
    
    return predict
# EVOLVE-BLOCK-END
