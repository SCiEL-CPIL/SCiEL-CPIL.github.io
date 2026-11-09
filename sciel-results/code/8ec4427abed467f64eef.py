"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_kernel(kind, x_train, x_test, **params):
    """Compute kernel matrix between x_train and x_test."""
    if kind == 'rbf':
        length_scale = params.get('length_scale', 1.0)
        diff = x_train[:, np.newaxis] - x_test[np.newaxis, :]
        return np.exp(-0.5 * (diff / length_scale) ** 2)
    elif kind == 'poly':
        degree = params.get('degree', 3)
        gamma = params.get('gamma', 1.0)
        coef0 = params.get('coef0', 1.0)
        return (gamma * np.outer(x_train, x_test) + coef0) ** degree
    elif kind == 'rational':
        gamma = params.get('gamma', 1.0)
        coef0 = params.get('coef0', 1.0)
        return 1.0 / (1.0 + gamma * np.abs(x_train[:, np.newaxis] - x_test[np.newaxis, :]))
    else:
        raise ValueError(f"Unknown kernel: {kind}")


def _fit_kernel_cv(x, y, kernel_kind, param_grid, n_folds=5, deadline=None):
    """Fit kernel regression with cross-validation over parameter grid.
    
    Returns (best_params, best_cv_score) or (None, inf) if timeout.
    """
    n = len(x)
    folds = np.array_split(np.arange(n), n_folds)
    best_cv = np.inf
    best_params = None
    
    for params in param_grid:
        if deadline is not None and time.time() > deadline:
            return None, np.inf
        
        cv_errors = []
        for k in range(n_folds):
            if deadline is not None and time.time() > deadline:
                return None, np.inf
            
            val_idx = folds[k]
            train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != k])
            
            x_tr = x[train_idx]
            y_tr = y[train_idx]
            x_val = x[val_idx]
            y_val = y[val_idx]
            
            try:
                K_train = _compute_kernel(kernel_kind, x_tr, x_tr, **params)
                K_test = _compute_kernel(kernel_kind, x_val, x_tr, **params)
                
                # Add small regularization for numerical stability
                reg = 1e-8 * np.eye(len(x_tr))
                K_train_reg = K_train + reg
                
                alpha = np.linalg.solve(K_train_reg, y_tr)
                pred = K_test @ alpha
                
                if not np.all(np.isfinite(pred)):
                    continue
                
                mse = np.mean((pred - y_val) ** 2)
                var_y = np.var(y_val)
                if var_y < 1e-15:
                    continue
                cv_errors.append(mse / var_y)
            except Exception:
                continue
        
        if cv_errors:
            mean_cv = np.mean(cv_errors)
            if mean_cv < best_cv:
                best_cv = mean_cv
                best_params = params
    
    return best_params, best_cv


def _make_kernel_predictor(kernel_kind, params, x_train, y_train):
    """Create a predictor using kernel regression."""
    K_train = _compute_kernel(kernel_kind, x_train, x_train, **params)
    reg = 1e-8 * np.eye(len(x_train))
    K_train_reg = K_train + reg
    
    try:
        alpha = np.linalg.solve(K_train_reg, y_train)
    except np.linalg.LinAlgError:
        alpha = np.linalg.lstsq(K_train_reg, y_train, rcond=None)[0]
    
    def predict(X):
        x_test = X[:, 0] if X.ndim == 2 and X.shape[1] == 1 else np.asarray(X).ravel()
        K_test = _compute_kernel(kernel_kind, x_test, x_train, **params)
        return K_test @ alpha
    
    return predict


def _make_poly_predictor(degree, x_train, y_train):
    """Create a polynomial predictor."""
    try:
        coeffs = np.polyfit(x_train, y_train, degree)
        def predict(X):
            x_test = X[:, 0] if X.ndim == 2 and X.shape[1] == 1 else np.asarray(X).ravel()
            return np.polyval(coeffs, x_test)
        return predict
    except Exception:
        return None


def _make_interp_predictor(x_train, y_train):
    """Create a cubic interpolation predictor."""
    try:
        from scipy.interpolate import CubicSpline
        cs = CubicSpline(x_train, y_train)
        def predict(X):
            x_test = X[:, 0] if X.ndim == 2 and X.shape[1] == 1 else np.asarray(X).ravel()
            return cs(x_test)
        return predict
    except Exception:
        return None


def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 285  # Safety margin before 300s limit
    
    X = X_train.astype(np.float64)
    y = y_train.astype(np.float64)
    n, d = X.shape
    
    x = X[:, 0].copy() if d == 1 else X.ravel()
    
    # Sort by x
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    
    x_range = x_sorted.max() - x_sorted.min()
    if x_range < 1e-10:
        x_range = 1.0
    
    best_predict = None
    best_score = np.inf
    
    def evaluate_predictor(predict_func, x_eval, y_eval):
        try:
            pred = predict_func(np.column_stack([x_eval]))
            if not np.all(np.isfinite(pred)):
                return np.inf
            mse = np.mean((pred - y_eval) ** 2)
            var_y = np.var(y_eval)
            if var_y < 1e-15:
                return np.inf
            return mse / var_y
        except Exception:
            return np.inf
    
    # Strategy 1: RBF kernel with multiple length scales
    rbf_grid = []
    for log_ls in np.arange(-1.5, 3.0, 0.3):
        ls = 10 ** log_ls * x_range / 10
        if 0 < ls < 20 * x_range:
            rbf_grid.append({'length_scale': ls})
    
    if rbf_grid:
        best_params, best_cv = _fit_kernel_cv(x_sorted, y_sorted, 'rbf', rbf_grid, 
                                               n_folds=5, deadline=deadline - 30)
        if best_params is not None and best_cv < np.inf:
            predict_func = _make_kernel_predictor('rbf', best_params, x_sorted, y_sorted)
            if predict_func is not None:
                train_score = evaluate_predictor(predict_func, x_sorted, y_sorted)
                if train_score < best_score:
                    best_score = train_score
                    best_predict = predict_func
    
    # Strategy 2: Polynomial kernel with various degrees and gammas
    poly_grid = []
    for degree in [2, 3, 4, 5]:
        for log_gamma in np.arange(-2, 2, 0.5):
            gamma = 10 ** log_gamma / x_range
            poly_grid.append({'degree': degree, 'gamma': gamma, 'coef0': 1.0})
    
    if poly_grid and time.time() < deadline - 25:
        best_params, best_cv = _fit_kernel_cv(x_sorted, y_sorted, 'poly', poly_grid,
                                               n_folds=5, deadline=deadline - 20)
        if best_params is not None and best_cv < np.inf:
            predict_func = _make_kernel_predictor('poly', best_params, x_sorted, y_sorted)
            if predict_func is not None:
                train_score = evaluate_predictor(predict_func, x_sorted, y_sorted)
                if train_score < best_score:
                    best_score = train_score
                    best_predict = predict_func
    
    # Strategy 3: Rational kernel
    rational_grid = []
    for log_gamma in np.arange(-2, 2, 0.5):
        gamma = 10 ** log_gamma / x_range
        rational_grid.append({'gamma': gamma, 'coef0': 1.0})
    
    if rational_grid and time.time() < deadline - 15:
        best_params, best_cv = _fit_kernel_cv(x_sorted, y_sorted, 'rational', rational_grid,
                                               n_folds=5, deadline=deadline - 10)
        if best_params is not None and best_cv < np.inf:
            predict_func = _make_kernel_predictor('rational', best_params, x_sorted, y_sorted)
            if predict_func is not None:
                train_score = evaluate_predictor(predict_func, x_sorted, y_sorted)
                if train_score < best_score:
                    best_score = train_score
                    best_predict = predict_func
    
    # Strategy 4: Polynomial regression (fast, good for smooth functions)
    if time.time() < deadline - 8:
        for degree in [2, 3, 4, 5, 6, 7, 8, 10]:
            if time.time() > deadline - 5:
                break
            predict_func = _make_poly_predictor(degree, x_sorted, y_sorted)
            if predict_func is not None:
                train_score = evaluate_predictor(predict_func, x_sorted, y_sorted)
                if train_score < best_score:
                    best_score = train_score
                    best_predict = predict_func
    
    # Strategy 5: Cubic spline interpolation
    if time.time() < deadline - 3:
        predict_func = _make_interp_predictor(x_sorted, y_sorted)
        if predict_func is not None:
            train_score = evaluate_predictor(predict_func, x_sorted, y_sorted)
            if train_score < best_score:
                best_score = train_score
                best_predict = predict_func
    
    # Final fallback: constant predictor
    if best_predict is None:
        mean_y = np.mean(y)
        best_predict = lambda X: np.full(np.asarray(X).shape[0], mean_y)
    
    return best_predict
# EVOLVE-BLOCK-END
