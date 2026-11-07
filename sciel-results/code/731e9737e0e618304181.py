import numpy as np
import time
import warnings
from scipy.interpolate import CubicSpline, UnivariateSpline
from scipy.optimize import minimize_scalar, minimize

warnings.filterwarnings('ignore')

_START_TIME = None
TIME_LIMIT = 330.0
SAFETY_MARGIN = 30.0

def _time_left():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)

def _rbf_kernel(X1, X2, length_scale):
    X1 = np.asarray(X1, dtype=np.float64).reshape(-1, 1)
    X2 = np.asarray(X2, dtype=np.float64).reshape(-1, 1)
    d2 = np.sum((X1 - X2.T)**2, axis=1)
    return np.exp(-0.5 * d2 / (length_scale**2))

def _fit_ridge_kernel(x_train, y_train, x_test, length_scale, alpha):
    """Fit ridge regression in RKHS induced by RBF kernel."""
    K = _rbf_kernel(x_train, x_train, length_scale)
    n = len(x_train)
    K_reg = K + alpha * np.eye(n)
    K_test = _rbf_kernel(x_test, x_train, length_scale)
    
    try:
        w = np.linalg.solve(K_reg, y_train)
        return K_test @ w
    except np.linalg.LinAlgError:
        w, _, _, _ = np.linalg.lstsq(K_reg, y_train, rcond=None)
        return K_test @ w

def _fit_rbf_kernel_ridge_cv(x, y, length_scales, alphas, n_folds=5):
    """Cross-validate RBF kernel + ridge, return best params and predictions."""
    n = len(y)
    np.random.seed(42)
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    best_mse = np.inf
    best_ls = length_scales[0]
    best_alpha = alphas[0]
    
    for ls in length_scales:
        if _time_left() < 5.0:
            break
        for alpha in alphas:
            if _time_left() < 5.0:
                break
            fold_mses = []
            for i in range(n_folds):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
                
                try:
                    pred_val = _fit_ridge_kernel(x[train_idx], y[train_idx], x[val_idx], ls, alpha)
                    mse = np.mean((pred_val - y[val_idx])**2)
                    fold_mses.append(mse)
                except Exception:
                    fold_mses.append(np.inf)
            
            mean_mse = np.mean(fold_mses)
            if mean_mse < best_mse:
                best_mse = mean_mse
                best_ls = ls
                best_alpha = alpha
    
    return best_ls, best_alpha, best_mse

def _build_chebyshev_basis(x, x_min, x_max, n_terms):
    """Build Chebyshev polynomial basis of degree 0 to n_terms-1."""
    x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
    T = np.zeros((len(x), n_terms))
    if n_terms > 0:
        T[:, 0] = 1.0
    if n_terms > 1:
        T[:, 1] = x_norm
    for k in range(2, n_terms):
        T[:, k] = 2 * x_norm * T[:, k-1] - T[:, k-2]
    return T

def _fit_chebyshev_ridge(x, y, x_test, n_terms, lam, x_min, x_max):
    """Fit Chebyshev polynomial + ridge."""
    V = _build_chebyshev_basis(x, x_min, x_max, n_terms)
    A = V.T @ V + lam * np.eye(n_terms)
    try:
        w = np.linalg.solve(A, V.T @ y)
    except np.linalg.LinAlgError:
        w, _, _, _ = np.linalg.lstsq(A, V.T @ y, rcond=None)
    V_test = _build_chebyshev_basis(x_test, x_min, x_max, n_terms)
    return V_test @ w

def _fit_chebyshev_ridge_cv(x, y, n_terms_options, lambdas, n_folds=5):
    """Cross-validate Chebyshev basis + ridge."""
    n = len(y)
    np.random.seed(42)
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    best_mse = np.inf
    best_n = n_terms_options[0]
    best_lambda = lambdas[0]
    x_min, x_max = np.min(x), np.max(x)
    
    for n_terms in n_terms_options:
        if _time_left() < 5.0:
            break
        for lam in lambdas:
            if _time_left() < 5.0:
                break
            fold_mses = []
            for i in range(n_folds):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
                
                try:
                    pred_val = _fit_chebyshev_ridge(x[train_idx], y[train_idx], x[val_idx], n_terms, lam, x_min, x_max)
                    mse = np.mean((pred_val - y[val_idx])**2)
                    fold_mses.append(mse)
                except Exception:
                    fold_mses.append(np.inf)
            
            mean_mse = np.mean(fold_mses)
            if mean_mse < best_mse:
                best_mse = mean_mse
                best_n = n_terms
                best_lambda = lam
    
    return best_n, best_lambda, best_mse

def _build_fourier_basis(x, x_min, x_max, n_terms):
    """Build Fourier basis: 1, cos(k*pi*x/L), sin(k*pi*x/L) for k=1.."""
    L = x_max - x_min
    if L <= 0:
        L = 1.0
    n = len(x)
    # n_terms total columns: 1 + 2*floor((n_terms-1)/2)
    F = np.zeros((n, n_terms))
    F[:, 0] = 1.0
    col = 1
    k = 1
    while col < n_terms:
        arg = k * np.pi * (x - x_min) / L
        F[:, col] = np.cos(arg)
        col += 1
        if col < n_terms:
            F[:, col] = np.sin(arg)
            col += 1
        k += 1
    return F

def _fit_fourier_ridge(x, y, x_test, n_terms, lam, x_min, x_max):
    """Fit Fourier basis + ridge."""
    F = _build_fourier_basis(x, x_min, x_max, n_terms)
    A = F.T @ F + lam * np.eye(n_terms)
    try:
        w = np.linalg.solve(A, F.T @ y)
    except np.linalg.LinAlgError:
        w, _, _, _ = np.linalg.lstsq(A, F.T @ y, rcond=None)
    F_test = _build_fourier_basis(x_test, x_min, x_max, n_terms)
    return F_test @ w

def _fit_fourier_ridge_cv(x, y, n_terms_options, lambdas, n_folds=5):
    """Cross-validate Fourier basis + ridge."""
    n = len(y)
    np.random.seed(42)
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    best_mse = np.inf
    best_n = n_terms_options[0]
    best_lambda = lambdas[0]
    x_min, x_max = np.min(x), np.max(x)
    
    for n_terms in n_terms_options:
        if _time_left() < 5.0:
            break
        for lam in lambdas:
            if _time_left() < 5.0:
                break
            fold_mses = []
            for i in range(n_folds):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
                
                try:
                    pred_val = _fit_fourier_ridge(x[train_idx], y[train_idx], x[val_idx], n_terms, lam, x_min, x_max)
                    mse = np.mean((pred_val - y[val_idx])**2)
                    fold_mses.append(mse)
                except Exception:
                    fold_mses.append(np.inf)
            
            mean_mse = np.mean(fold_mses)
            if mean_mse < best_mse:
                best_mse = mean_mse
                best_n = n_terms
                best_lambda = lam
    
    return best_n, best_lambda, best_mse

def _spline_fit(x, y, x_test, s):
    """Fit a smoothing spline."""
    try:
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]
        
        if s <= 0:
            s = 1e-8 * max(np.var(y), 1e-10)
        
        spl = UnivariateSpline(x_sorted, y_sorted, k=3, s=s)
        return spl(x_test)
    except Exception:
        return np.full(len(x_test), np.mean(y))

def _spline_cv(x, y, s_options, n_folds=5):
    """Cross-validate smoothing spline."""
    n = len(y)
    np.random.seed(42)
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    best_mse = np.inf
    best_s = s_options[0]
    
    for s in s_options:
        if _time_left() < 5.0:
            break
        fold_mses = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            
            try:
                pred_val = _spline_fit(x[train_idx], y[train_idx], x[val_idx], s)
                mse = np.mean((pred_val - y[val_idx])**2)
                fold_mses.append(mse)
            except Exception:
                fold_mses.append(np.inf)
        
        mean_mse = np.mean(fold_mses)
        if mean_mse < best_mse:
            best_mse = mean_mse
            best_s = s
    
    return best_s, best_mse

def _fit_cubic_spline(x, y, x_test):
    """Fit an interpolating cubic spline (no smoothing)."""
    try:
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]
        spl = CubicSpline(x_sorted, y_sorted)
        return spl(x_test)
    except Exception:
        return np.full(len(x_test), np.mean(y))

def _fit_loess(x_train, y_train, x_test, frac, degree=1):
    """LOESS prediction."""
    x_train = np.asarray(x_train, dtype=np.float64).ravel()
    y_train = np.asarray(y_train, dtype=np.float64).ravel()
    x_test = np.asarray(x_test, dtype=np.float64).ravel()
    
    n = len(x_train)
    m = max(2, int(frac * n))
    preds = np.empty(len(x_test))
    
    for i, xv in enumerate(x_test):
        if _time_left() < 0.5:
            preds[i:] = np.mean(y_train)
            break
        
        dists = np.abs(x_train - xv)
        if m >= n:
            idx = np.arange(n)
        else:
            idx = np.argpartition(dists, m - 1)[:m]
        
        max_dist = dists[idx].max()
        if max_dist < 1e-15:
            max_dist = 1e-10
        w = (1 - (dists[idx] / max_dist)**3)**3
        
        # Weighted polynomial fit
        try:
            if degree == 0:
                preds[i] = np.sum(w * y_train[idx]) / np.sum(w)
            else:
                Xw = x_train[idx] - xv
                W = np.diag(w)
                A = np.vstack([np.ones_like(Xw), Xw]).T
                WA = W @ A
                Wb = W @ y_train[idx]
                try:
                    coef = np.linalg.solve(WA.T @ A + 1e-12 * np.eye(2), WA.T @ y_train[idx])
                    preds[i] = coef[0]
                except np.linalg.LinAlgError:
                    preds[i] = np.sum(w * y_train[idx]) / np.sum(w)
        except Exception:
            preds[i] = np.mean(y_train)
    
    return preds

def _loess_cv(x, y, frac_options, n_folds=5):
    """Cross-validate LOESS."""
    n = len(y)
    np.random.seed(42)
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    best_mse = np.inf
    best_frac = frac_options[0]
    
    for frac in frac_options:
        if _time_left() < 5.0:
            break
        fold_mses = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            
            try:
                pred_val = _fit_loess(x[train_idx], y[train_idx], x[val_idx], frac)
                mse = np.mean((pred_val - y[val_idx])**2)
                fold_mses.append(mse)
            except Exception:
                fold_mses.append(np.inf)
        
        mean_mse = np.mean(fold_mses)
        if mean_mse < best_mse:
            best_mse = mean_mse
            best_frac = frac
    
    return best_frac, best_mse

def fit(X_train, y_train):
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    X = np.asarray(X_train, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y_train, dtype=np.float64).reshape(-1)
    n = len(y)
    x = X[:, 0].copy()
    
    x_min, x_max = np.min(x), np.max(x)
    
    # Method 1: RBF kernel + ridge
    ls_options = np.logspace(-2, 1, 8)
    alpha_options = np.logspace(-4, 1, 8)
    best_ls, best_alpha, kernel_mse = _fit_rbf_kernel_ridge_cv(x, y, ls_options, alpha_options)
    
    # Method 2: Chebyshev basis + ridge
    n_terms_options = [5, 8, 12, 16, 20, 25, 30]
    lam_options = np.logspace(-4, 2, 10)
    best_n_cheb, best_lam_cheb, cheb_mse = _fit_chebyshev_ridge_cv(x, y, n_terms_options, lam_options)
    
    # Method 3: Fourier basis + ridge
    n_terms_fourier = [5, 8, 12, 16, 20, 25]
    best_n_four, best_lam_four, four_mse = _fit_fourier_ridge_cv(x, y, n_terms_fourier, lam_options)
    
    # Method 4: Smoothing spline
    s_options = np.logspace(-10, -2, 15)
    best_s, spline_mse = _spline_cv(x, y, s_options)
    
    # Method 5: LOESS
    frac_options = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    best_frac, loess_mse = _loess_cv(x, y, frac_options)
    
    # Method 6: Interpolating cubic spline
    try:
        # Quick LOOCV-like estimate using leave-5-out
        np.random.seed(42)
        indices = np.arange(n)
        np.random.shuffle(indices)
        folds = np.array_split(indices, 5)
        fold_mses = []
        for i in range(5):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
            pred_val = _fit_cubic_spline(x[train_idx], y[train_idx], x[val_idx])
            mse = np.mean((pred_val - y[val_idx])**2)
            fold_mses.append(mse)
        spline_interp_mse = np.mean(fold_mses)
    except Exception:
        spline_interp_mse = np.inf
    
    # Pick the best method
    methods = [
        ('kernel', kernel_mse, best_ls, best_alpha),
        ('chebyshev', cheb_mse, best_n_cheb, best_lam_cheb),
        ('fourier', four_mse, best_n_four, best_lam_four),
        ('spline', spline_mse, best_s, None),
        ('loess', loess_mse, best_frac, None),
        ('spline_interp', spline_interp_mse, None, None)
    ]
    methods.sort(key=lambda m: m[1])
    
    best_method = methods[0][0]
    
    # Build the final model
    if best_method == 'kernel':
        K = _rbf_kernel(x, x, best_ls)
        K_reg = K + best_alpha * np.eye(n)
        try:
            w = np.linalg.solve(K_reg, y)
        except np.linalg.LinAlgError:
            w, _, _, _ = np.linalg.lstsq(K_reg, y, rcond=None)
        
        x_store = x.copy()
        w_store = w.copy()
        ls_store = best_ls
        
        def predict(X):
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            xv = X_arr[:, 0]
            K_test = _rbf_kernel(xv, x_store, ls_store)
            return K_test @ w_store
    
    elif best_method == 'chebyshev':
        V = _build_chebyshev_basis(x, x_min, x_max, best_n_cheb)
        A = V.T @ V + best_lam_cheb * np.eye(best_n_cheb)
        try:
            w = np.linalg.solve(A, V.T @ y)
        except np.linalg.LinAlgError:
            w, _, _, _ = np.linalg.lstsq(A, V.T @ y, rcond=None)
        
        w_store = w.copy()
        xmin_store = x_min
        xmax_store = x_max
        n_store = best_n_cheb
        
        def predict(X):
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            xv = X_arr[:, 0]
            V_test = _build_chebyshev_basis(xv, xmin_store, xmax_store, n_store)
            return V_test @ w_store
    
    elif best_method == 'fourier':
        F = _build_fourier_basis(x, x_min, x_max, best_n_four)
        A = F.T @ F + best_lam_four * np.eye(best_n_four)
        try:
            w = np.linalg.solve(A, F.T @ y)
        except np.linalg.LinAlgError:
            w, _, _, _ = np.linalg.lstsq(A, F.T @ y, rcond=None)
        
        w_store = w.copy()
        xmin_store = x_min
        xmax_store = x_max
        n_store = best_n_four
        
        def predict(X):
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            xv = X_arr[:, 0]
            F_test = _build_fourier_basis(xv, xmin_store, xmax_store, n_store)
            return F_test @ w_store
    
    elif best_method == 'spline':
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]
        try:
            spl = UnivariateSpline(x_sorted, y_sorted, k=3, s=best_s)
            
            def predict(X):
                X_arr = np.asarray(X, dtype=np.float64)
                if X_arr.ndim == 1:
                    X_arr = X_arr.reshape(-1, 1)
                xv = X_arr[:, 0]
                return spl(xv)
        except Exception:
            mean_y = np.mean(y)
            def predict(X):
                X_arr = np.asarray(X, dtype=np.float64)
                if X_arr.ndim == 1:
                    X_arr = X_arr.reshape(-1, 1)
                return np.full(len(X_arr), mean_y)
    
    elif best_method == 'loess':
        x_store = x.copy()
        y_store = y.copy()
        frac_store = best_frac
        
        def predict(X):
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            xv = X_arr[:, 0]
            return _fit_loess(x_store, y_store, xv, frac_store)
    
    else:  # spline_interp
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]
        try:
            spl = CubicSpline(x_sorted, y_sorted)
            
            def predict(X):
                X_arr = np.asarray(X, dtype=np.float64)
                if X_arr.ndim == 1:
                    X_arr = X_arr.reshape(-1, 1)
                xv = X_arr[:, 0]
                return spl(xv)
        except Exception:
            mean_y = np.mean(y)
            def predict(X):
                X_arr = np.asarray(X, dtype=np.float64)
                if X_arr.ndim == 1:
                    X_arr = X_arr.reshape(-1, 1)
                return np.full(len(X_arr), mean_y)
    
    return predict
