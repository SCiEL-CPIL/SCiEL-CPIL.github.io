"""Symbolic regression over 10 hidden equations: Multi-stage oracle with GP, Poly-Kernel, Rational."""
# EVOLVE-BLOCK-START
import time
import numpy as np
import scipy.linalg
from scipy.optimize import minimize

_START = time.time()
_DEADLINE = 300.0

def _remaining():
    return _DEADLINE - (time.time() - _START)

def _make_predict_rbf(X_train, y_train, gamma, alpha):
    diff = X_train[:, np.newaxis, :] - X_train[np.newaxis, :, :]
    sq_dists = np.sum(diff**2, axis=-1)
    K = np.exp(-gamma * sq_dists)
    K_reg = K + alpha * np.eye(len(X_train))
    try:
        w = scipy.linalg.solve(K_reg, y_train)
    except Exception:
        w = np.linalg.pinv(K_reg) @ y_train
        
    def predict(X):
        diff_test = X[:, np.newaxis, :] - X_train[np.newaxis, :, :]
        sq_dists_test = np.sum(diff_test**2, axis=-1)
        K_test = np.exp(-gamma * sq_dists_test)
        return K_test @ w
    return predict

def _make_predict_linear(basis_func, w, offset=None):
    def predict(X):
        A = basis_func(X)
        if offset is not None:
            return A @ w + offset
        return A @ w
    return predict

def _make_predict_rational(num_basis, den_basis, w_num, w_den, c_den):
    def predict(X):
        A_num = num_basis(X)
        A_den = den_basis(X)
        num = A_num @ w_num
        den = A_den @ w_den + c_den
        # Prevent division by zero
        mask = np.abs(den) < 1e-8
        den[mask] = 1e-8
        return num / den
    return predict

def _basis_poly2(X):
    d = X.shape[1]
    cols = [np.ones(len(X))]
    for j in range(d):
        cols.append(X[:, j])
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
    return np.column_stack(cols)

def _basis_poly3(X):
    d = X.shape[1]
    cols = [np.ones(len(X))]
    for j in range(d):
        cols.append(X[:, j])
        cols.append(X[:, j] ** 2)
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
    for i in range(d):
        cols.append(X[:, i] ** 3)
    for i in range(d):
        for j in range(i, d):
            for k in range(j, d):
                cols.append(X[:, i] * X[:, j] * X[:, k])
    return np.column_stack(cols)

def _basis_poly4(X):
    d = X.shape[1]
    cols = [np.ones(len(X))]
    for j in range(d):
        cols.append(X[:, j])
        cols.append(X[:, j] ** 2)
        cols.append(X[:, j] ** 3)
        cols.append(X[:, j] ** 4)
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
            cols.append(X[:, i] ** 2 * X[:, j])
            cols.append(X[:, i] * X[:, j] ** 2)
            cols.append(X[:, i] ** 2 * X[:, j] ** 2)
    for i in range(d):
        for j in range(i, d):
            for k in range(j, d):
                cols.append(X[:, i] * X[:, j] * X[:, k])
    return np.column_stack(cols)

def _basis_rich(X):
    d = X.shape[1]
    cols = [np.ones(len(X))]
    for j in range(d):
        cols.append(X[:, j])
        cols.append(X[:, j] ** 2)
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
            
    for j in range(d):
        x = X[:, j]
        cols.append(np.sin(x))
        cols.append(np.cos(x))
        cols.append(np.tanh(x))
        cols.append(np.arctan(x))
        cols.append(np.log(x))
        cols.append(np.sqrt(x))
        cols.append(np.abs(x))
        cols.append(np.exp(-x))
        cols.append(np.sin(x * x))
        cols.append(np.cos(x * x))
        
    if d >= 2:
        for i in range(d):
            for j in range(i+1, d):
                cols.append(np.sin(X[:, i]) * X[:, j])
                cols.append(np.tanh(X[:, i]) * X[:, j])
                cols.append(np.log(X[:, i]) * X[:, j])
                
    return np.column_stack(cols)

def _basis_trig_poly(X):
    d = X.shape[1]
    cols = [np.ones(len(X))]
    for j in range(d):
        cols.append(X[:, j])
        cols.append(np.sin(X[:, j]))
        cols.append(np.cos(X[:, j]))
        cols.append(np.tanh(X[:, j]))
        cols.append(np.arctan(X[:, j]))
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
            cols.append(np.sin(X[:, i]) * np.cos(X[:, j]))
            cols.append(np.sin(X[:, i] + X[:, j]))
            cols.append(np.cos(X[:, i] + X[:, j]))
    return np.column_stack(cols)

def _basis_comprehensive(X):
    d = X.shape[1]
    cols = [np.ones(len(X))]
    
    for j in range(d):
        cols.append(X[:, j])
        cols.append(X[:, j] ** 2)
        cols.append(X[:, j] ** 3)
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
            cols.append(X[:, i] ** 2 * X[:, j])
            cols.append(X[:, i] * X[:, j] ** 2)
    for i in range(d):
        for j in range(i, d):
            for k in range(j, d):
                cols.append(X[:, i] * X[:, j] * X[:, k])
    
    for j in range(d):
        x = X[:, j]
        cols.append(np.sin(x))
        cols.append(np.cos(x))
        cols.append(np.tanh(x))
        cols.append(np.arctan(x))
        cols.append(np.log(x))
        cols.append(np.sqrt(x))
        cols.append(np.exp(-x))
        cols.append(np.sin(x**2))
        cols.append(np.cos(x**2))
        cols.append(np.tanh(x**2))
        
    if d >= 2:
        for i in range(d):
            for j in range(i+1, min(i+2, d)):
                cols.append(np.sin(X[:, i]) * X[:, j])
                cols.append(np.tanh(X[:, i]) * X[:, j])
                cols.append(np.log(X[:, i]) * X[:, j])
                cols.append(X[:, i] * np.sin(X[:, j]))
                cols.append(X[:, i] * np.tanh(X[:, j]))
                
    return np.column_stack(cols)

_CANDIDATE_BASES = [
    _basis_poly2,
    _basis_poly3,
    _basis_poly4,
    _basis_rich,
    _basis_trig_poly,
    _basis_comprehensive
]

def _try_exact_fit(basis_func, X, y, tol=1e-6):
    try:
        A = basis_func(X)
        if not np.all(np.isfinite(A)):
            return None
        w, residuals, rank, s = np.linalg.lstsq(A, y, rcond=None)
        pred = A @ w
        if not np.all(np.isfinite(pred)):
            return None
        mse = np.mean((pred - y)**2)
        scale = max(1.0, np.max(np.abs(y)))
        if mse <= (tol * scale)**2:
            return w, mse
    except Exception:
        pass
    return None

def _cv_rbf_score(X, y, gamma, alpha, n_folds=5):
    n = len(y)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // n_folds
    cv_errors = []
    
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        
        if len(train_idx) < 10:
            continue
            
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        try:
            pred = _make_predict_rbf(X_train_fold, y_train_fold, gamma, alpha)(X_val_fold)
            err = np.mean((pred - y_val_fold)**2)
            if np.isfinite(err):
                cv_errors.append(err)
        except Exception:
            continue
    
    if cv_errors:
        return np.mean(cv_errors)
    return float('inf')

def _poly_kernel_predict(X_train, y_train, degree, coef0, alpha):
    """Polynomial Kernel Ridge Regression."""
    def kernel(X1, X2):
        return (X1 @ X2.T + coef0) ** degree
    K = kernel(X_train, X_train)
    K_reg = K + alpha * np.eye(len(X_train))
    try:
        w = scipy.linalg.solve(K_reg, y_train)
    except Exception:
        w = np.linalg.pinv(K_reg) @ y_train
        
    def predict(X):
        K_test = kernel(X, X_train)
        return K_test @ w
    return predict

def _cv_poly_kernel_score(X, y, degree, coef0, alpha, n_folds=5):
    n = len(y)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // n_folds
    cv_errors = []
    
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        
        if len(train_idx) < 10:
            continue
            
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        try:
            pred = _poly_kernel_predict(X_train_fold, y_train_fold, degree, coef0, alpha)(X_val_fold)
            err = np.mean((pred - y_val_fold)**2)
            if np.isfinite(err):
                cv_errors.append(err)
        except Exception:
            continue
    
    if cv_errors:
        return np.mean(cv_errors)
    return float('inf')

def _matern52_kernel(X1, X2, length_scale, variance):
    sq_dists = np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :])**2, axis=-1)
    return variance * (1 + np.sqrt(3 * sq_dists / length_scale**2)) * np.exp(-np.sqrt(3 * sq_dists / length_scale**2))

def _gp_predict(X_train, y_train, length_scale, variance, noise, alpha_reg=1e-6):
    """Gaussian Process Regression with Matern-5/2 kernel."""
    K = _matern52_kernel(X_train, X_train, length_scale, variance)
    K_reg = K + (noise + alpha_reg) * np.eye(len(X_train))
    try:
        L = scipy.linalg.cholesky(K_reg, lower=True)
        alpha_vec = scipy.linalg.cho_solve((L, True), y_train)
    except Exception:
        try:
            alpha_vec = scipy.linalg.solve(K_reg, y_train)
        except Exception:
            alpha_vec = np.linalg.pinv(K_reg) @ y_train
            
    def predict(X):
        K_s = _matern52_kernel(X, X_train, length_scale, variance)
        return K_s @ alpha_vec
    return predict

def _cv_gp_score(X, y, length_scale, variance, noise, n_folds=5):
    n = len(y)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // n_folds
    cv_errors = []
    
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        
        if len(train_idx) < 10:
            continue
            
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        try:
            pred = _gp_predict(X_train_fold, y_train_fold, length_scale, variance, noise)(X_val_fold)
            err = np.mean((pred - y_val_fold)**2)
            if np.isfinite(err):
                cv_errors.append(err)
        except Exception:
            continue
    
    if cv_errors:
        return np.mean(cv_errors)
    return float('inf')

def _fit_rational(X, y, num_basis, den_basis):
    """Fit rational function y = num(X)/den(X) using nonlinear least squares."""
    A_num = num_basis(X)
    A_den = den_basis(X)
    
    # Initial guess: linear fit on numerator, denominator = 1
    w_num_init, _, _, _ = np.linalg.lstsq(A_num, y, rcond=None)
    w_den_init = np.zeros(A_den.shape[1])
    w_den_init[0] = 1.0  # Constant term in denominator
    c_den_init = 1.0
    
    params_init = np.concatenate([w_num_init, w_den_init, [c_den_init]])
    
    def objective(params):
        w_num = params[:A_num.shape[1]]
        w_den = params[A_num.shape[1]:A_num.shape[1]+A_den.shape[1]]
        c_den = params[-1]
        num = A_num @ w_num
        den = A_den @ w_den + c_den
        mask = np.abs(den) < 1e-6
        den[mask] = 1e-6
        residual = num / den - y
        return np.sum(residual**2)
    
    try:
        result = minimize(objective, params_init, method='Nelder-Mead', 
                         options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
        if result.success or np.isfinite(result.fun):
            best_params = result.x
            w_num_best = best_params[:A_num.shape[1]]
            w_den_best = best_params[A_num.shape[1]:A_num.shape[1]+A_den.shape[1]]
            c_den_best = best_params[-1]
            return w_num_best, w_den_best, c_den_best
    except Exception:
        pass
    return None

def _cv_rational_score(X, y, num_basis, den_basis, n_folds=5):
    n = len(y)
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_size = n // n_folds
    cv_errors = []
    
    for fold in range(n_folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n
        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
        
        if len(train_idx) < 10:
            continue
            
        X_train_fold = X[train_idx]
        y_train_fold = y[train_idx]
        X_val_fold = X[val_idx]
        y_val_fold = y[val_idx]
        
        try:
            result = _fit_rational(X_train_fold, y_train_fold, num_basis, den_basis)
            if result is not None:
                w_num, w_den, c_den = result
                pred = _make_predict_rational(num_basis, den_basis, w_num, w_den, c_den)(X_val_fold)
                err = np.mean((pred - y_val_fold)**2)
                if np.isfinite(err):
                    cv_errors.append(err)
        except Exception:
            continue
    
    if cv_errors:
        return np.mean(cv_errors)
    return float('inf')

def fit(X_train, y_train):
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = y.reshape(-1)
    
    n_samples = len(X)
    d = X.shape[1]
    
    # 1. Try Exact Fit with various bases
    best_exact_predict = None
    best_exact_err = float('inf')
    
    for basis_func in _CANDIDATE_BASES:
        if _remaining() < 15.0:
            break
        try:
            result = _try_exact_fit(basis_func, X, y)
            if result is not None:
                w, err = result
                if err < best_exact_err:
                    best_exact_err = err
                    best_exact_predict = _make_predict_linear(basis_func, w)
        except Exception:
            continue
            
    if best_exact_predict is not None:
        return best_exact_predict
        
    # 2. Multi-stage Oracle Selection
    candidates = []
    
    # Stage A: RBF Kernel Ridge Regression
    best_rbf_predict = None
    best_rbf_cv_err = float('inf')
    gammas = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    alphas = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
    
    for gamma in gammas:
        if _remaining() < 10.0:
            break
        for alpha in alphas:
            if _remaining() < 5.0:
                break
            try:
                cv_err = _cv_rbf_score(X, y, gamma, alpha)
                if cv_err < best_rbf_cv_err:
                    best_rbf_cv_err = cv_err
                    best_rbf_predict = _make_predict_rbf(X, y, gamma, alpha)
            except Exception:
                continue
    
    if best_rbf_predict is not None:
        candidates.append((best_rbf_cv_err, best_rbf_predict))
        
    # Stage B: Polynomial Kernel Ridge Regression
    best_polyk_predict = None
    best_polyk_cv_err = float('inf')
    degrees = [2, 3, 4]
    coef0s = [1.0, 5.0]
    poly_alphas = [1e-3, 1e-2, 0.1, 1.0]
    
    for degree in degrees:
        if _remaining() < 10.0:
            break
        for coef0 in coef0s:
            if _remaining() < 5.0:
                break
            for alpha in poly_alphas:
                if _remaining() < 3.0:
                    break
                try:
                    cv_err = _cv_poly_kernel_score(X, y, degree, coef0, alpha)
                    if cv_err < best_polyk_cv_err:
                        best_polyk_cv_err = cv_err
                        best_polyk_predict = _poly_kernel_predict(X, y, degree, coef0, alpha)
                except Exception:
                    continue
    
    if best_polyk_predict is not None:
        candidates.append((best_polyk_cv_err, best_polyk_predict))
        
    # Stage C: Gaussian Process with Matern-5/2
    best_gp_predict = None
    best_gp_cv_err = float('inf')
    # Heuristic initialization for length scale based on data range
    x_range = np.max(X, axis=0) - np.min(X, axis=0)
    x_range = x_range[x_range > 0]
    if len(x_range) > 0:
        typical_scale = np.median(x_range)
    else:
        typical_scale = 1.0
        
    length_scales = [0.5 * typical_scale, typical_scale, 2.0 * typical_scale]
    variances = [0.5, 1.0, 2.0]
    noises = [1e-4, 1e-3, 1e-2]
    
    for ls in length_scales:
        if _remaining() < 10.0:
            break
        for var in variances:
            if _remaining() < 5.0:
                break
            for noise in noises:
                if _remaining() < 3.0:
                    break
                try:
                    cv_err = _cv_gp_score(X, y, ls, var, noise)
                    if cv_err < best_gp_cv_err:
                        best_gp_cv_err = cv_err
                        best_gp_predict = _gp_predict(X, y, ls, var, noise)
                except Exception:
                    continue
    
    if best_gp_predict is not None:
        candidates.append((best_gp_cv_err, best_gp_predict))
        
    # Stage D: Rational Function Approximator
    best_rat_predict = None
    best_rat_cv_err = float('inf')
    # Use simple bases for rational function
    num_basis = _basis_poly2
    den_basis = _basis_poly2
    
    try:
        cv_err = _cv_rational_score(X, y, num_basis, den_basis)
        if np.isfinite(cv_err):
            best_rat_cv_err = cv_err
            result = _fit_rational(X, y, num_basis, den_basis)
            if result is not None:
                w_num, w_den, c_den = result
                best_rat_predict = _make_predict_rational(num_basis, den_basis, w_num, w_den, c_den)
    except Exception:
        pass
    
    if best_rat_predict is not None:
        candidates.append((best_rat_cv_err, best_rat_predict))
        
    # Select best candidate
    if candidates:
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]
        
    # 3. Ultimate Fallback: Ridge regression on poly2
    try:
        A = _basis_poly2(X)
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        return _make_predict_linear(_basis_poly2, w)
    except Exception:
        mean_y = np.mean(y)
        return lambda X: np.full(len(X), mean_y)

# EVOLVE-BLOCK-END
