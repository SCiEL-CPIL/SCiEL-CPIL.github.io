import numpy as np
from scipy.optimize import curve_fit, least_squares, minimize
from scipy.linalg import cho_solve, cho_factor
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 290.0  # Safety margin before 300s limit
    
    X_flat = X_train.flatten().astype(float)
    y_flat = y_train.flatten().astype(float)
    n = len(X_flat)
    
    # Handle constant case
    if np.var(y_flat) < 1e-12:
        mean_y = np.mean(y_flat)
        def predict_const(X):
            return np.full(np.asarray(X).shape, mean_y)
        return predict_const
    
    # Normalize X and Y for better conditioning
    X_mean = np.mean(X_flat)
    X_std = np.std(X_flat)
    if X_std < 1e-10:
        X_std = 1.0
    X_norm = (X_flat - X_mean) / X_std
    
    y_mean = np.mean(y_flat)
    y_std = np.std(y_flat)
    if y_std < 1e-10:
        y_std = 1.0
    y_norm = (y_flat - y_mean) / y_std
    
    # Quick polynomial check: try degrees 1-6
    best_poly_resid = np.inf
    best_poly_coef = None
    for deg in range(1, 7):
        if time.time() - start_time > deadline - 30:
            break
        try:
            coef = np.polyfit(X_flat, y_flat, deg)
            y_pred = np.polyval(coef, X_flat)
            resid = np.mean((y_pred - y_flat)**2)
            if resid < best_poly_resid:
                best_poly_resid = resid
                best_poly_coef = coef
            if resid < 1e-10:
                break
        except Exception:
            continue
    
    # If polynomial fits very well, use it
    if best_poly_coef is not None and best_poly_resid < 1e-8:
        def predict_poly(X):
            x = np.asarray(X).flatten()
            return np.polyval(best_poly_coef, x).reshape(np.asarray(X).shape)
        return predict_poly
    
    # Gaussian Process Regression with composite kernel
    # We'll use a simple GP implementation with RBF + Periodic + Linear kernels
    
    # Kernel parameters to optimize:
    # RBF: length_scale_rbf, variance_rbf
    # Periodic: period, length_scale_per
    # Linear: variance_linear
    # Noise: variance_noise
    
    def composite_kernel(X1, X2, params):
        """Composite kernel: RBF + Periodic + Linear"""
        ls_rbf, var_rbf, period, ls_per, var_per, var_lin, var_noise = params
        
        # RBF (Matern 5/2 or standard RBF)
        d2_rbf = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
        d2_rbf = np.maximum(d2_rbf, 0)
        K_rbf = var_rbf * np.exp(-0.5 * d2_rbf / (ls_rbf ** 2))
        
        # Periodic kernel
        d2_per = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
        K_per = var_per * np.cos(np.pi * d2_per / period) * np.exp(-0.5 * d2_per / (ls_per ** 2))
        
        # Linear kernel
        K_lin = var_lin * (X1 @ X2.T)
        
        # Add noise to diagonal
        K = K_rbf + K_per + K_lin
        if np.any(np.isclose(X1[:, 0], X2[:, 0])):
            K[np.arange(len(X1)), np.arange(len(X2))] += var_noise
        
        return K
    
    def log_likelihood(params):
        """Negative log marginal likelihood"""
        ls_rbf, var_rbf, period, ls_per, var_per, var_lin, var_noise = params
        
        # Ensure positive values
        ls_rbf = max(ls_rbf, 1e-3)
        var_rbf = max(var_rbf, 1e-3)
        period = max(period, 0.1)
        ls_per = max(ls_per, 1e-3)
        var_per = max(var_per, 1e-3)
        var_lin = max(var_lin, 1e-3)
        var_noise = max(var_noise, 1e-6)
        
        K = composite_kernel(X_norm[:, None], X_norm[:, None], 
                             [ls_rbf, var_rbf, period, ls_per, var_per, var_lin, var_noise])
        
        try:
            L = cho_factor(K + 1e-10 * np.eye(n))
            alpha = cho_solve(L, y_norm)
            log_det = 2 * np.sum(np.log(np.diag(cho_factor(K + 1e-10 * np.eye(n))[0])))
            nll = 0.5 * y_norm @ alpha + 0.5 * log_det + 0.5 * n * np.log(2 * np.pi)
            return nll
        except Exception:
            return 1e10
    
    # Optimize kernel parameters
    # Initial guess: reasonable defaults
    x_range = np.ptp(X_norm)
    initial_params = [
        1.0,      # ls_rbf
        1.0,      # var_rbf
        2.0,      # period
        1.0,      # ls_per
        0.1,      # var_per
        0.1,      # var_lin
        0.01      # var_noise
    ]
    
    best_params = initial_params
    best_nll = 1e10
    
    # Try a few different initializations
    inits = [
        [1.0, 1.0, 2.0, 1.0, 0.1, 0.1, 0.01],
        [0.5, 1.0, 1.0, 0.5, 0.5, 0.5, 0.01],
        [2.0, 1.0, 4.0, 2.0, 0.1, 0.1, 0.01],
        [1.0, 1.0, 0.5, 0.5, 1.0, 0.1, 0.01],
    ]
    
    for p0 in inits:
        if time.time() - start_time > deadline - 10:
            break
        try:
            result = minimize(log_likelihood, p0, method='L-BFGS-B', 
                            bounds=[(1e-3, 10), (1e-3, 10), (0.1, 10), (1e-3, 10), (1e-3, 10), (1e-3, 10), (1e-6, 1)],
                            options={'maxiter': 100, 'ftol': 1e-6})
            if result.fun < best_nll:
                best_nll = result.fun
                best_params = result.x
        except Exception:
            continue
    
    # Build final GP model with best parameters
    ls_rbf, var_rbf, period, ls_per, var_per, var_lin, var_noise = best_params
    ls_rbf = max(ls_rbf, 1e-3)
    var_rbf = max(var_rbf, 1e-3)
    period = max(period, 0.1)
    ls_per = max(ls_per, 1e-3)
    var_per = max(var_per, 1e-3)
    var_lin = max(var_lin, 1e-3)
    var_noise = max(var_noise, 1e-6)
    
    K_train = composite_kernel(X_norm[:, None], X_norm[:, None], 
                               [ls_rbf, var_rbf, period, ls_per, var_per, var_lin, var_noise])
    K_train += 1e-10 * np.eye(n)
    
    try:
        L = cho_factor(K_train)
        alpha = cho_solve(L, y_norm)
        
        def predict_gp(X):
            x = np.asarray(X).flatten()
            x_norm = (x - X_mean) / X_std
            K_star = composite_kernel(x_norm[:, None], X_norm[:, None], 
                                      [ls_rbf, var_rbf, period, ls_per, var_per, var_lin, var_noise])
            mu_norm = K_star @ alpha
            mu = y_mean + y_std * mu_norm
            return mu.reshape(np.asarray(X).shape)
        
        # Verify GP prediction quality on training data
        y_pred_train = predict_gp(X_train)
        train_mse = np.mean((y_pred_train - y_flat)**2)
        
        # If GP is good, use it
        if train_mse < 1e-4 * np.var(y_flat) + 1e-10:
            return predict_gp
            
    except Exception:
        pass
    
    # Fallback: high-degree polynomial
    fallback_deg = min(15, n - 1)
    try:
        fallback_coef = np.polyfit(X_flat, y_flat, fallback_deg)
    except Exception:
        fallback_coef = np.polyfit(X_flat, y_flat, 5)
    
    def predict_fallback(X):
        x = np.asarray(X).flatten()
        return np.polyval(fallback_coef, x).reshape(np.asarray(X).shape)
    
    return predict_fallback
