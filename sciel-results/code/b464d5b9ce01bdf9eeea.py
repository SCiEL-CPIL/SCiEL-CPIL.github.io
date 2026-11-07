"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.interpolate import PchipInterpolator, CubicSpline, make_lsq_spline
from scipy.optimize import minimize
import time


def _pchip_predict(X, y):
    X_sorted, y_sorted = X, y
    idx = np.argsort(X_sorted)
    X_s = X_sorted[idx]
    y_s = y_sorted[idx]
    interp = PchipInterpolator(X_s, y_s)
    return lambda X_test: interp(X_test)


def _cubic_spline_predict(X, y):
    X_sorted, y_sorted = X, y
    idx = np.argsort(X_sorted)
    X_s = X_sorted[idx]
    y_s = y_sorted[idx]
    interp = CubicSpline(X_s, y_s)
    return lambda X_test: interp(X_test)


def _gaussian_process_fit(X, y):
    """
    Gaussian Process regression with Matern 5/2 kernel.
    Optimize hyperparameters via negative log marginal likelihood.
    """
    X_sorted, y_sorted = X, y
    idx = np.argsort(X_sorted)
    X_s = X_sorted[idx]
    y_s = y_sorted[idx]
    
    # Normalize X and Y for numerical stability
    X_mean = np.mean(X_s)
    X_std = np.std(X_s) + 1e-8
    y_mean = np.mean(y_s)
    y_std = np.std(y_s) + 1e-8
    
    X_norm = (X_s - X_mean) / X_std
    y_norm = (y_s - y_mean) / y_std
    
    def matern52_kernel(X1, X2, l, sigma):
        d = np.abs(X1[:, None] - X2[None, :])
        sqrt_3_d = np.sqrt(3.0) * d
        return sigma ** 2 * (1.0 + sqrt_3_d) * np.exp(-sqrt_3_d)
    
    def negative_log_mll(params):
        log_l, log_sigma, log_noise = params
        l = np.exp(log_l)
        sigma = np.exp(log_sigma)
        noise = np.exp(log_noise)
        
        K = matern52_kernel(X_norm, X_norm, l, sigma) + noise * np.eye(len(X_norm))
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            nll = 0.5 * y_norm @ alpha + 0.5 * log_det + 0.5 * len(y_norm) * np.log(2 * np.pi)
            return nll
        except np.linalg.LinAlgError:
            return 1e10
    
    # Initial guess
    x0 = np.array([0.0, np.log(np.std(y_norm) + 1e-8), np.log(1e-8)])
    
    result = minimize(negative_log_mll, x0, method='L-BFGS-B', bounds=[(-4, 4), (-4, 4), (-10, 2)])
    
    if not result.success:
        # Fallback to reasonable defaults
        l = 0.5
        sigma = np.std(y_norm)
        noise = 1e-8
    else:
        log_l, log_sigma, log_noise = result.x
        l = np.exp(log_l)
        sigma = np.exp(log_sigma)
        noise = np.exp(log_noise)
    
    K_train = matern52_kernel(X_norm, X_norm, l, sigma) + noise * np.eye(len(X_norm))
    L = np.linalg.cholesky(K_train)
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
    
    def predict(X_test):
        X_test_norm = (X_test - X_mean) / X_std
        K_test = matern52_kernel(X_test_norm, X_norm, l, sigma)
        y_pred_norm = K_test @ alpha
        y_pred = y_pred_norm * y_std + y_mean
        return y_pred
    
    return predict


def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 250.0  # Safety margin
    
    # All input dimensions are 1D, so X_train is (200, 1)
    X = X_train[:, 0]
    y = y_train
    
    # Sort by X
    idx = np.argsort(X)
    X_s = X[idx]
    y_s = y[idx]
    
    # Check for monotonicity
    dy = np.diff(y_s)
    is_monotone_increasing = np.all(dy >= -1e-10)
    is_monotone_decreasing = np.all(dy <= 1e-10)
    is_monotone = is_monotone_increasing or is_monotone_decreasing
    
    # Candidate 1: PCHIP (if monotone)
    if is_monotone:
        try:
            pchip_pred = _pchip_predict(X, y)
            # Evaluate on training points
            train_pred = pchip_pred(X)
            pchip_mse = np.mean((train_pred - y) ** 2)
        except:
            pchip_pred = None
            pchip_mse = np.inf
    else:
        pchip_pred = None
        pchip_mse = np.inf
    
    # Candidate 2: Cubic Spline
    try:
        spline_pred = _cubic_spline_predict(X, y)
        train_pred = spline_pred(X)
        spline_mse = np.mean((train_pred - y) ** 2)
    except:
        spline_pred = None
        spline_mse = np.inf
    
    # Candidate 3: GP (if time allows)
    gp_pred = None
    gp_mse = np.inf
    if time.time() < deadline - 50.0:
        try:
            gp_pred = _gaussian_process_fit(X, y)
            train_pred = gp_pred(X)
            gp_mse = np.mean((train_pred - y) ** 2)
        except:
            gp_pred = None
            gp_mse = np.inf
    
    # Choose best based on training MSE (since data is noiseless, lower MSE is better)
    candidates = []
    if pchip_pred is not None:
        candidates.append((pchip_mse, pchip_pred))
    if spline_pred is not None:
        candidates.append((spline_mse, spline_pred))
    if gp_pred is not None:
        candidates.append((gp_mse, gp_pred))
    
    if not candidates:
        # Fallback: linear interpolation
        return lambda X_test: np.interp(X_test, X, y)
    
    candidates.sort(key=lambda x: x[0])
    best_mse, best_pred = candidates[0]
    
    # Final check: ensure predictions are finite
    def safe_predict(X_test):
        preds = best_pred(X_test)
        return np.where(np.isfinite(preds), preds, 0.0)
    
    return safe_predict
# EVOLVE-BLOCK-END
