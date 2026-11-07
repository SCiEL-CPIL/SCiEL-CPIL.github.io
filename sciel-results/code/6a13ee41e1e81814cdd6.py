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

def _loocv_ridge_fast(basis, y, lambdas):
    """LOOCV for ridge regression using the hat matrix trick."""
    n = len(y)
    best_lambda = lambdas[0]
    best_mse = np.inf
    
    for lam in lambdas:
        A = basis.T @ basis + lam * np.eye(basis.shape[1])
        try:
            A_inv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            continue
        H = basis @ A_inv @ basis.T
        diag = np.diag(H)
        
        # LOOCV residuals
        fitted = basis @ (A_inv @ (basis.T @ y))
        residuals = (y - fitted) / (1.0 - diag)
        mse = np.mean(residuals**2)
        
        if mse < best_mse:
            best_mse = mse
            best_lambda = lam
    
    return best_lambda, best_mse

def _chebyshev_terms(x, n):
    """Compute Chebyshev polynomial values T_0(x), T_1(x), ..., T_{n-1}(x)."""
    x = np.asarray(x, dtype=np.float64)
    if n <= 0:
        return np.empty((len(x), 0))
    if n == 1:
        return np.ones((len(x), 1))
    if n == 2:
        return np.stack([np.ones_like(x), x], axis=1)
    
    T = np.empty((len(x), n))
    T[:, 0] = 1.0
    T[:, 1] = x
    for k in range(2, n):
        T[:, k] = 2.0 * x * T[:, k-1] - T[:, k-2]
    return T

def _build_rich_basis(x, n_cheb, n_fourier, x_min, x_max):
    """Build a rich basis: Chebyshev + Fourier terms."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    
    # Chebyshev basis (map x to [-1, 1])
    if x_max > x_min:
        x_norm = (x - x_min) / (x_max - x_min) * 2.0 - 1.0
    else:
        x_norm = np.zeros_like(x)
    cheb = _chebyshev_terms(x_norm, n_cheb)
    
    # Fourier basis
    if n_fourier > 0:
        freqs = np.arange(1, n_fourier + 1)
        # Use period based on range
        period = x_max - x_min if x_max > x_min else 1.0
        angles = 2.0 * np.pi * np.outer(x, freqs) / period
        sin_terms = np.sin(angles)
        cos_terms = np.cos(angles)
        fourier = np.hstack([sin_terms, cos_terms])
    else:
        fourier = np.empty((n, 0))
    
    return np.hstack([cheb, fourier])

def _loess_predict(x_train, y_train, x_test, frac, deg):
    """
    LOESS prediction using local polynomial regression.
    For each test point, fit a weighted polynomial using nearby training points.
    """
    x_train = np.asarray(x_train, dtype=np.float64).reshape(-1)
    y_train = np.asarray(y_train, dtype=np.float64).reshape(-1)
    x_test = np.asarray(x_test, dtype=np.float64).reshape(-1)
    
    n_train = len(x_train)
    n_test = len(x_test)
    if n_train == 0:
        return np.zeros(n_test)
    
    # Number of neighbors to use
    k = max(3, int(frac * n_train))
    k = min(k, n_train)
    
    # Tricube weights
    def tricube(u):
        return (1 - u**3)**3
    
    predictions = np.zeros(n_test)
    
    for i in range(n_test):
        # Find k nearest neighbors
        dists = np.abs(x_test[i] - x_train)
        # Get indices of k nearest
        if k < n_train:
            idx = np.argpartition(dists, k)[:k]
        else:
            idx = np.arange(n_train)
        
        # Compute weights
        max_dist = np.max(dists[idx])
        if max_dist < 1e-12:
            w = np.ones(len(idx))
        else:
            u = dists[idx] / max_dist
            w = tricube(u)
        
        # Fit local polynomial
        x_loc = x_train[idx]
        y_loc = y_train[idx]
        w_loc = w
        
        # Weighted least squares
        V = np.vander(x_loc - x_test[i], N=deg+1, increasing=True)
        W = np.diag(w_loc)
        
        try:
            # Solve (V^T W V) beta = V^T W y
            VtW = V.T * w_loc[:, None]
            A = VtW @ V
            b = VtW @ y_loc
            
            # Add small regularization for stability
            A += 1e-8 * np.eye(deg+1)
            
            beta = np.linalg.solve(A, b)
            predictions[i] = beta[0]
        except np.linalg.LinAlgError:
            # Fallback to weighted mean
            predictions[i] = np.sum(w_loc * y_loc) / np.sum(w_loc)
    
    return predictions

def _loess_cv(x, y, frac_options, deg_options, n_folds=5):
    """Cross-validate LOESS parameters."""
    n = len(y)
    np.random.seed(42)
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    best_mse = np.inf
    best_frac = frac_options[0]
    best_deg = deg_options[0]
    
    for frac in frac_options:
        if _time_left() < 10.0:
            break
        for deg in deg_options:
            if _time_left() < 10.0:
                break
            fold_mses = []
            for i in range(n_folds):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
                
                try:
                    pred_val = _loess_predict(x[train_idx], y[train_idx], x[val_idx], frac, deg)
                    mse = np.mean((pred_val - y[val_idx])**2)
                    fold_mses.append(mse)
                except Exception:
                    fold_mses.append(np.inf)
            
            mean_mse = np.mean(fold_mses)
            if mean_mse < best_mse:
                best_mse = mean_mse
                best_frac = frac
                best_deg = deg
    
    return best_frac, best_deg, best_mse

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
    x_range = x_max - x_min if x_max > x_min else 1.0
    
    # Store results from each method
    results = []
    
    # Method 1: Rich basis (Chebyshev + Fourier) with ridge
    if _time_left() > 60.0:
        best_basis_score = np.inf
        best_n_cheb = 10
        best_n_fourier = 0
        best_weights = None
        best_lambda = 1e-3
        
        n_cheb_options = [6, 10, 15, 20, 25]
        n_fourier_options = [0, 5, 10]
        lambda_options = np.logspace(-4, 2, 10)
        
        for n_cheb in n_cheb_options:
            if _time_left() < 30.0:
                break
            for n_fourier in n_fourier_options:
                if _time_left() < 30.0:
                    break
                try:
                    basis = _build_rich_basis(x, n_cheb, n_fourier, x_min, x_max)
                    if basis.shape[1] >= n:
                        continue
                    
                    best_lambda, best_mse = _loocv_ridge_fast(basis, y, lambda_options)
                    
                    if best_mse < best_basis_score:
                        A = basis.T @ basis + best_lambda * np.eye(basis.shape[1])
                        try:
                            weights = np.linalg.solve(A, basis.T @ y)
                        except np.linalg.LinAlgError:
                            weights = np.linalg.lstsq(A, basis.T @ y, rcond=None)[0]
                        
                        best_basis_score = best_mse
                        best_n_cheb = n_cheb
                        best_n_fourier = n_fourier
                        best_weights = weights.copy()
                except Exception:
                    continue
        
        if best_weights is not None:
            results.append(('basis', best_basis_score, best_n_cheb, best_n_fourier, best_weights))
    
    # Method 2: RBF kernel + ridge
    if _time_left() > 60.0:
        ls_options = np.logspace(-2, 1, 6)
        alpha_options = np.logspace(-4, 1, 6)
        
        np.random.seed(42)
        indices = np.arange(n)
        np.random.shuffle(indices)
        folds = np.array_split(indices, 5)
        
        best_kernel_mse = np.inf
        best_ls = ls_options[0]
        best_alpha = alpha_options[0]
        
        for ls in ls_options:
            if _time_left() < 30.0:
                break
            for alpha in alpha_options:
                if _time_left() < 30.0:
                    break
                fold_mses = []
                for i in range(5):
                    val_idx = folds[i]
                    train_idx = np.concatenate([folds[j] for j in range(5) if j != i])
                    
                    try:
                        K_tr = _rbf_kernel(x[train_idx], x[train_idx], ls)
                        K_tr[np.diag_indices_from(K_tr)] += alpha
                        K_val = _rbf_kernel(x[val_idx], x[train_idx], ls)
                        
                        try:
                            w = np.linalg.solve(K_tr, y[train_idx])
                        except np.linalg.LinAlgError:
                            w = np.linalg.lstsq(K_tr, y[train_idx], rcond=None)[0]
                        
                        pred_val = K_val @ w
                        mse = np.mean((pred_val - y[val_idx])**2)
                        fold_mses.append(mse)
                    except Exception:
                        fold_mses.append(np.inf)
                
                mean_mse = np.mean(fold_mses)
                if mean_mse < best_kernel_mse:
                    best_kernel_mse = mean_mse
                    best_ls = ls
                    best_alpha = alpha
        
        if best_kernel_mse < np.inf:
            results.append(('kernel', best_kernel_mse, best_ls, best_alpha, None))
    
    # Method 3: LOESS
    if _time_left() > 30.0:
        frac_options = [0.1, 0.2, 0.3, 0.4, 0.5]
        deg_options = [1, 2, 3]
        
        best_frac, best_deg, loess_mse = _loess_cv(x, y, frac_options, deg_options)
        
        if loess_mse < np.inf:
            results.append(('loess', loess_mse, best_frac, best_deg, None))
    
    # Pick the best method
    if not results:
        # Fallback: constant prediction
        mean_y = np.mean(y)
        def predict(X):
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            return np.full(len(X_arr), mean_y)
        return predict
    
    results.sort(key=lambda r: r[1])
    best_method = results[0][0]
    
    # Build the final model
    if best_method == 'basis':
        _, _, n_cheb, n_fourier, weights = results[0]
        
        def predict(X):
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            xv = X_arr[:, 0]
            basis_test = _build_rich_basis(xv, n_cheb, n_fourier, x_min, x_max)
            return basis_test @ weights
        return predict
    
    elif best_method == 'kernel':
        _, _, ls, alpha, _ = results[0]
        
        K = _rbf_kernel(x, x, ls)
        K[np.diag_indices_from(K)] += alpha
        try:
            w = np.linalg.solve(K, y)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(K, y, rcond=None)[0]
        
        x_store = x.copy()
        w_store = w.copy()
        ls_store = ls
        
        def predict(X):
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            xv = X_arr[:, 0]
            K_test = _rbf_kernel(xv, x_store, ls_store)
            return K_test @ w_store
        return predict
    
    elif best_method == 'loess':
        _, _, frac, deg, _ = results[0]
        
        x_store = x.copy()
        y_store = y.copy()
        frac_store = frac
        deg_store = deg
        
        def predict(X):
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            xv = X_arr[:, 0]
            return _loess_predict(x_store, y_store, xv, frac_store, deg_store)
        return predict
    
    else:
        # Should not reach here, but fallback
        mean_y = np.mean(y)
        def predict(X):
            X_arr = np.asarray(X, dtype=np.float64)
            if X_arr.ndim == 1:
                X_arr = X_arr.reshape(-1, 1)
            return np.full(len(X_arr), mean_y)
        return predict
