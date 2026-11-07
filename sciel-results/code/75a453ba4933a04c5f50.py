"""
Strategy: Gaussian Process Regression with Matérn 5/2 kernel.
Hyperparameters are selected via 5-fold cross-validation to minimize validation MSE.
This avoids the overfitting issues of high-degree polynomials and neural networks
on small datasets, and provides a smooth, flexible fit for arbitrary 1D functions.
"""
import numpy as np
import time
import warnings
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")

_START_TIME = None
TIME_LIMIT = 300.0
SAFETY_MARGIN = 15.0

def _time_left():
    global _START_TIME
    if _START_TIME is None:
        return TIME_LIMIT - SAFETY_MARGIN
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - _START_TIME)

def _matern52_kernel(X1, X2, length_scale, signal_var):
    """
    Compute the Matérn 5/2 kernel matrix between two sets of points.
    K(x, x') = signal_var * (1 + sqrt(5)*r + 5/3*r^2) * exp(-sqrt(5)*r)
    where r = ||x - x'|| / length_scale
    """
    # Distance matrix
    dists = cdist(X1, X2)
    r = dists / length_scale
    sqrt5_r = np.sqrt(5.0) * r
    K = signal_var * (1.0 + sqrt5_r + (5.0/3.0) * r**2) * np.exp(-sqrt5_r)
    return K

def _fit_gp(X, y, time_budget):
    """
    Fit a Gaussian Process with Matérn 5/2 kernel.
    Returns a predict function.
    """
    n = len(y)
    if n < 2:
        return lambda X_in: np.full(len(X_in), y.mean())
    
    # Normalize inputs to [0, 1] or standardize? Standardization is safer for length scale interpretation.
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-8] = 1.0
    X_norm = (X - X_mean) / X_std
    
    y_mean = y.mean()
    y_std = y.std()
    y_std[y_std < 1e-8] = 1.0
    y_norm = (y - y_mean) / y_std
    
    # Initial hyperparameters
    # length_scale in normalized space. 1.0 is a reasonable start.
    # signal_var is the variance of the function. 1.0 is a reasonable start for normalized y.
    # noise_var is small since data is noiseless, but we need it for numerical stability.
    init_params = np.array([1.0, 1.0, 1e-6]) # [log_length_scale, log_signal_var, log_noise_var]
    
    def _neg_log_likelihood(params):
        log_ls = params[0]
        log_sv = params[1]
        log_nv = params[2]
        
        length_scale = np.exp(log_ls)
        signal_var = np.exp(log_sv)
        noise_var = np.exp(log_nv)
        
        try:
            K = _matern52_kernel(X_norm, X_norm, length_scale, signal_var)
            K[np.diag_indices_from(K)] += noise_var
            
            # Add jitter for numerical stability
            K += 1e-8 * np.eye(n)
            
            # Cholesky decomposition
            L = np.linalg.cholesky(K)
            
            # Solve L * alpha = y
            alpha = np.linalg.solve(L, y_norm)
            alpha = np.linalg.solve(L.T, alpha)
            
            # Marginal log likelihood
            ll = -0.5 * np.dot(y_norm, alpha) - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)
            
            if not np.isfinite(ll):
                return 1e10
            return -ll
        except:
            return 1e10
    
    # 5-Fold Cross Validation
    best_params = None
    best_cv_score = np.inf
    k = 5
    
    # Shuffle indices
    indices = np.arange(n)
    np.random.seed(42) # Fixed seed for reproducibility in CV split
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    
    for i in range(k):
        if _time_left() < 10:
            break
            
        val_idx = folds[i]
        train_idx = np.setdiff1d(indices, val_idx)
        
        X_train = X_norm[train_idx]
        y_train = y_norm[train_idx]
        X_val = X_norm[val_idx]
        y_val = y_norm[val_idx]
        
        # Optimize hyperparameters on train fold
        # Use Nelder-Mead or BFGS? Nelder-Mead is robust for small param spaces.
        try:
            res = minimize(_neg_log_likelihood, init_params, method='Nelder-Mead', 
                           options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 200})
            opt_params = res.x
        except:
            opt_params = init_params
        
        # Evaluate on validation fold
        log_ls, log_sv, log_nv = opt_params
        length_scale = np.exp(log_ls)
        signal_var = np.exp(log_sv)
        noise_var = np.exp(log_nv)
        
        K_train = _matern52_kernel(X_train, X_train, length_scale, signal_var)
        K_train[np.diag_indices_from(K_train)] += noise_var
        K_train += 1e-8 * np.eye(len(X_train))
        
        try:
            L = np.linalg.cholesky(K_train)
            alpha = np.linalg.solve(L, y_train)
            alpha = np.linalg.solve(L.T, alpha)
            
            K_val_train = _matern52_kernel(X_val, X_train, length_scale, signal_var)
            y_pred = K_val_train @ alpha
            
            mse = np.mean((y_pred - y_val)**2)
            
            if mse < best_cv_score:
                best_cv_score = mse
                best_params = opt_params
        except:
            continue
    
    # If CV failed, use the initial params or the best from a single full-data optimization
    if best_params is None:
        try:
            res = minimize(_neg_log_likelihood, init_params, method='Nelder-Mead', 
                           options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 200})
            best_params = res.x
        except:
            best_params = init_params
            
    # Final fit on all data with best params
    log_ls, log_sv, log_nv = best_params
    length_scale = np.exp(log_ls)
    signal_var = np.exp(log_sv)
    noise_var = np.exp(log_nv)
    
    K = _matern52_kernel(X_norm, X_norm, length_scale, signal_var)
    K[np.diag_indices_from(K)] += noise_var
    K += 1e-8 * np.eye(n)
    
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L, y_norm)
        alpha = np.linalg.solve(L.T, alpha)
    except:
        # Fallback to linear regression if GP fails
        X_ext = np.column_stack([np.ones(n), X_norm])
        w, *_ = np.linalg.lstsq(X_ext, y_norm, rcond=None)
        def predict_fallback(X_in):
            Xn_in = (X_in - X_mean) / X_std
            X_ext_in = np.column_stack([np.ones(len(Xn_in)), Xn_in])
            return (X_ext_in @ w) * y_std + y_mean
        return predict_fallback

    def predict(X_in):
        Xn_in = (X_in - X_mean) / X_std
        K_star = _matern52_kernel(Xn_in, X_norm, length_scale, signal_var)
        y_pred_norm = K_star @ alpha
        return y_pred_norm * y_std + y_mean

    return predict

def _fit_poly(X, y, degree):
    """Fallback polynomial regression."""
    x = X[:, 0]
    Xf = np.vander(x, N=degree+1, increasing=True)
    w, *_ = np.linalg.lstsq(Xf, y, rcond=None)
    return lambda X_in: np.vander(X_in[:, 0], N=degree+1, increasing=True) @ w

def fit(X_train, y_train):
    global _START_TIME
    _START_TIME = time.time()
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    
    # Ensure finite
    mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    X, y = X[mask], y[mask]
    
    if len(X) < 5:
        return lambda X_in: np.full(len(X_in), y.mean())
        
    # Try GP first
    try:
        if _time_left() > 30:
            predict_fn = _fit_gp(X, y, _time_left())
            # Check if prediction is valid
            test_pred = predict_fn(X)
            if np.all(np.isfinite(test_pred)):
                return predict_fn
    except Exception as e:
        pass
    
    # Fallback to polynomial if GP fails or is too slow
    try:
        if _time_left() > 10:
            # Try degree 3 and 5
            p3 = _fit_poly(X, y, 3)
            p5 = _fit_poly(X, y, 5)
            mse3 = np.mean((p3(X) - y)**2)
            mse5 = np.mean((p5(X) - y)**2)
            if mse3 < mse5:
                return p3
            else:
                return p5
    except:
        pass
        
    # Final fallback: constant
    y_mean = y.mean()
    return lambda X_in: np.full(len(X_in), y_mean)
