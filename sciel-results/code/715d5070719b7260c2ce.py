import numpy as np
import time
import warnings

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
        # Solve (K + alpha*I) w = y
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

def _fit_basis_ridge_cv(x, y, n_terms_options, lambdas, n_folds=5):
    """Cross-validate polynomial basis + ridge."""
    n = len(y)
    np.random.seed(42)
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    best_mse = np.inf
    best_n = n_terms_options[0]
    best_lambda = lambdas[0]
    x_min, x_max = np.min(x), np.max(x)
    x_range = x_max - x_min if x_max > x_min else 1.0
    
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
                    # Build Vandermonde-like basis for training
                    x_tr = x[train_idx]
                    x_tr_norm = (x_tr - x_min) / x_range
                    # Use powers up to n_terms-1
                    V = np.vander(x_tr_norm, N=n_terms, increasing=True)
                    A = V.T @ V + lam * np.eye(n_terms)
                    w = np.linalg.solve(A, V.T @ y[train_idx])
                    
                    # Predict on validation
                    x_va = x[val_idx]
                    x_va_norm = (x_va - x_min) / x_range
                    V_va = np.vander(x_va_norm, N=n_terms, increasing=True)
                    pred_va = V_va @ w
                    mse = np.mean((pred_va - y[val_idx])**2)
                    fold_mses.append(mse)
                except Exception:
                    fold_mses.append(np.inf)
            
            mean_mse = np.mean(fold_mses)
            if mean_mse < best_mse:
                best_mse = mean_mse
                best_n = n_terms
                best_lambda = lam
    
    return best_n, best_lambda, best_mse

def _spline_fit(x, y, x_test, n_knots, s=0.0):
    """Fit a smoothing spline."""
    try:
        from scipy.interpolate import UnivariateSpline
        # Sort by x
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]
        
        # Choose s based on n_knots
        if s == 0.0:
            s = 1e-6 * np.var(y)
        
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
                pred_val = _spline_fit(x[train_idx], y[train_idx], x[val_idx], 0, s)
                mse = np.mean((pred_val - y[val_idx])**2)
                fold_mses.append(mse)
            except Exception:
                fold_mses.append(np.inf)
        
        mean_mse = np.mean(fold_mses)
        if mean_mse < best_mse:
            best_mse = mean_mse
            best_s = s
    
    return best_s, best_mse

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
    
    # Method 1: RBF kernel + ridge
    ls_options = np.logspace(-2, 1, 8)
    alpha_options = np.logspace(-4, 1, 8)
    best_ls, best_alpha, kernel_mse = _fit_rbf_kernel_ridge_cv(x, y, ls_options, alpha_options)
    
    # Method 2: Polynomial basis + ridge
    n_terms_options = [5, 8, 12, 16, 20, 25]
    lam_options = np.logspace(-4, 2, 10)
    best_n, best_lam, poly_mse = _fit_basis_ridge_cv(x, y, n_terms_options, lam_options)
    
    # Method 3: Smoothing spline
    s_options = np.logspace(-8, -2, 12)
    best_s, spline_mse = _spline_cv(x, y, s_options)
    
    # Pick the best method
    methods = [
        ('kernel', kernel_mse, best_ls, best_alpha),
        ('poly', poly_mse, best_n, best_lam),
        ('spline', spline_mse, best_s, None)
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
    
    elif best_method == 'poly':
        x_min, x_max = np.min(x), np.max(x)
        x_range = x_max - x_min if x_max > x_min else 1.0
        V = np.vander((x - x_min) / x_range, N=best_n, increasing=True)
        A = V.T @ V + best_lam * np.eye(best_n)
        try:
            w = np.linalg.solve(A, V.T @ y)
        except np.linalg.LinAlgError:
            w, _, _, _ = np.linalg.lstsq(A, V.T @ y, rcond=None)
        
        w_store = w.copy()
        xmin_store = x_min
        xrange_store = x_range
        n_store = best_n
        
        def predict(X):
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            xv = X_arr[:, 0]
            xv_norm = (xv - xmin_store) / xrange_store
            V_test = np.vander(xv_norm, N=n_store, increasing=True)
            return V_test @ w_store
    
    else:  # spline
        order = np.argsort(x)
        x_sorted = x[order]
        y_sorted = y[order]
        try:
            from scipy.interpolate import UnivariateSpline
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
    
    return predict
