import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 300.0 - 5.0  # Safety margin

    # X_train is (N, 1)
    x = X_train[:, 0]
    y = y_train
    n = len(x)

    # Standardize x for stable GP hyperparameter optimization
    x_mean = np.mean(x)
    x_std = np.std(x)
    if x_std < 1e-8:
        x_std = 1.0
    x_norm = (x - x_mean) / x_std

    # Handle constant y
    if np.allclose(y, y[0]):
        y_val = y[0]
        return lambda X: np.full(len(X), y_val)

    y_std = np.std(y)
    if y_std < 1e-8:
        y_std = 1.0

    # GP Kernel: RBF
    # K(x, x') = sigma_f^2 * exp(-0.5 * (x - x')^2 / l^2)
    
    def log_likelihood(params):
        log_l, log_sigma_f = params
        l = np.exp(log_l)
        sigma_f = np.exp(log_sigma_f)
        
        # Compute kernel matrix
        diff = x_norm[:, None] - x_norm[None, :]
        K = sigma_f**2 * np.exp(-0.5 * diff**2 / l**2)
        
        # Add jitter for numerical stability
        K += 1e-8 * np.eye(n)
        
        # Cholesky decomposition
        try:
            L, lower = cho_factor(K)
        except:
            return 1e6 # Bad likelihood
            
        # Log determinant
        log_det = 2 * np.sum(np.log(np.diag(L)))
        
        # Solve K alpha = y
        try:
            alpha = cho_solve((L, lower), y)
        except:
            return 1e6
            
        # NLML
        nlml = 0.5 * y.dot(alpha) + 0.5 * log_det + 0.5 * n * np.log(2 * np.pi)
        return nlml
        
    # Initial guess for hyperparameters
    # l ~ 1, sigma_f ~ std(y)
    x0 = [np.log(1.0), np.log(y_std)]
    
    best_params = x0
    best_nll = np.inf
    
    # Try a few initial points to avoid local minima
    for l_init in [0.1, 0.5, 1.0, 2.0, 5.0]:
        x0_try = [np.log(l_init), np.log(y_std)]
        
        # Check time
        if time.time() > deadline:
            break
            
        try:
            res = minimize(log_likelihood, x0_try, method='Nelder-Mead', 
                           options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
            if res.fun < best_nll:
                best_nll = res.fun
                best_params = res.x
        except Exception:
            pass

    log_l, log_sigma_f = best_params
    l = np.exp(log_l)
    sigma_f = np.exp(log_sigma_f)
    
    # Precompute for prediction
    diff = x_norm[:, None] - x_norm[None, :]
    K = sigma_f**2 * np.exp(-0.5 * diff**2 / l**2)
    K += 1e-8 * np.eye(n)
    
    try:
        L, lower = cho_factor(K)
        alpha = cho_solve((L, lower), y)
    except Exception:
        # Fallback: Ridge Regression with high degree polynomial
        # If GP fails, use a safe fallback
        deg = 5
        X_poly = np.vander(x, N=deg+1, increasing=True)
        X_poly_norm = (X_poly - np.mean(X_poly, axis=0)) / (np.std(X_poly, axis=0) + 1e-8)
        lam = 1.0
        w = np.linalg.solve(X_poly_norm.T @ X_poly_norm + lam * np.eye(deg+1), X_poly_norm.T @ y)
        
        def predict_fallback(X):
            x_test = X[:, 0]
            X_test_poly = np.vander(x_test, N=deg+1, increasing=True)
            X_test_poly_norm = (X_test_poly - np.mean(X_poly, axis=0)) / (np.std(X_poly, axis=0) + 1e-8)
            return X_test_poly_norm @ w

        return predict_fallback

    def predict(X):
        x_test = X[:, 0]
        x_test_norm = (x_test - x_mean) / x_std
        
        # K_* = K(x_test, x_train)
        diff_test = x_test_norm[:, None] - x_norm[None, :]
        K_star = sigma_f**2 * np.exp(-0.5 * diff_test**2 / l**2)
        
        # Prediction mean
        y_pred = K_star.dot(alpha)
        
        return y_pred

    return predict
