"""Symbolic regression using Gaussian Process with adaptive kernel."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _gp_kernel(X1, X2, length_scale, variance):
    """Compute RBF kernel matrix."""
    # Squared Euclidean distance
    sq_dists = np.sum(X1**2, axis=1)[:, np.newaxis] + \
               np.sum(X2**2, axis=1)[np.newaxis, :] - \
               2.0 * X1 @ X2.T
    sq_dists = np.maximum(sq_dists, 0)
    return variance * np.exp(-0.5 * sq_dists / length_scale**2)

def _gp_marginal_likelihood(params, X, y, noise_var):
    """Negative log marginal likelihood for GP with RBF kernel."""
    length_scale = np.exp(params[0])
    variance = np.exp(params[1])
    
    K = _gp_kernel(X, X, length_scale, variance)
    K += noise_var * np.eye(len(X))
    
    try:
        L = np.linalg.cholesky(K + 1e-10 * np.eye(len(X)))
    except np.linalg.LinAlgError:
        return 1e10
    
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    log_det = 2.0 * np.sum(np.log(np.diag(L)))
    
    n = len(y)
    ml = -0.5 * y.T @ alpha - 0.5 * log_det - 0.5 * n * np.log(2 * np.pi)
    return -ml

def _gp_predict(X_train, y_train, X_test, length_scale, variance, noise_var):
    """Predict using GP."""
    K_train = _gp_kernel(X_train, X_train, length_scale, variance)
    K_train += noise_var * np.eye(len(X_train))
    K_test = _gp_kernel(X_test, X_train, length_scale, variance)
    
    try:
        L = np.linalg.cholesky(K_train + 1e-10 * np.eye(len(K_train)))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        v = np.linalg.solve(L, K_test.T)
        mean = K_test @ alpha
    except np.linalg.LinAlgError:
        # Fallback: use linear regression
        Xb = np.hstack([X_train, np.ones((len(X_train), 1))])
        try:
            w = np.linalg.lstsq(Xb, y_train, rcond=None)[0]
            Xb_test = np.hstack([X_test, np.ones((len(X_test), 1))])
            mean = Xb_test @ w
        except Exception:
            mean = np.full(len(X_test), y_train.mean())
    
    return mean

def fit(X_train, y_train):
    """Fit a Gaussian Process regression model."""
    start_time = time.time()
    deadline = start_time + 270  # Safety margin
    
    X = np.asarray(X_train, dtype=float)
    y = np.asarray(y_train, dtype=float)
    n, d = X.shape
    
    # Normalize inputs
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-10
    X_norm = (X - X_mean) / X_std
    
    # Normalize outputs for better GP conditioning
    y_mean = y.mean()
    y_std = y.std() + 1e-10
    y_norm = (y - y_mean) / y_std
    
    # Estimate noise level from data
    # Use a simple heuristic: assume noise is small for noiseless data
    noise_var = 1e-4
    
    # Optimize GP hyperparameters
    # Initial guesses: length_scale ~ 1, variance ~ 1
    initial_params = np.array([0.0, 0.0])
    
    best_params = initial_params.copy()
    best_mll = np.inf
    
    # Try multiple initializations
    inits = [
        np.array([-1.0, 0.0]),   # length_scale ~ 0.37
        np.array([0.0, 0.0]),    # length_scale ~ 1.0
        np.array([1.0, 0.0]),    # length_scale ~ 2.72
        np.array([0.5, 0.5]),    # length_scale ~ 1.65, variance ~ 1.65
        np.array([-0.5, 1.0]),   # length_scale ~ 0.61, variance ~ 2.72
    ]
    
    for init in inits:
        if time.time() > deadline - 30:
            break
        try:
            result = minimize(
                _gp_marginal_likelihood,
                init,
                args=(X_norm, y_norm, noise_var),
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4}
            )
            if result.fun < best_mll:
                best_mll = result.fun
                best_params = result.x
        except Exception:
            continue
    
    length_scale = np.exp(best_params[0])
    variance = np.exp(best_params[1])
    
    # Fit final GP
    try:
        K_train = _gp_kernel(X_norm, X_norm, length_scale, variance)
        K_train += noise_var * np.eye(n)
        L = np.linalg.cholesky(K_train + 1e-10 * np.eye(n))
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
        use_gp = True
    except np.linalg.LinAlgError:
        use_gp = False
    
    # Fallback: linear regression
    if not use_gp:
        Xb = np.hstack([X_norm, np.ones((n, 1))])
        try:
            w = np.linalg.lstsq(Xb, y, rcond=None)[0]
        except Exception:
            w = np.zeros(d + 1)
            w[-1] = y.mean()
        
        def predict(X_test):
            X_test = np.asarray(X_test, dtype=float)
            X_test_norm = (X_test - X_mean) / X_std
            Xb_test = np.hstack([X_test_norm, np.ones((len(X_test), 1))])
            pred = Xb_test @ w
            pred = np.where(np.isfinite(pred), pred, y_mean)
            return pred
        return predict
    
    # Create predict function
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=float)
        X_test_norm = (X_test - X_mean) / X_std
        
        K_test = _gp_kernel(X_test_norm, X_norm, length_scale, variance)
        
        try:
            v = np.linalg.solve(L, K_test.T)
            mean_norm = K_test @ alpha
            mean = mean_norm * y_std + y_mean
        except Exception:
            # Fallback prediction
            mean = np.full(len(X_test), y_mean)
        
        mean = np.where(np.isfinite(mean), mean, y_mean)
        return mean
    
    return predict
# EVOLVE-BLOCK-END
