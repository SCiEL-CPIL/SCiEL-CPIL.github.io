"""Symbolic regression using GP with robust fallbacks and multi-start optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
import warnings

def fit(X_train, y_train):
    t_start = time.time()
    DEADLINE = 305.0  # Safety margin before 330s kill

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)
        
    n, d = X.shape
    y_flat = y.flatten()

    # Standardize X and y
    mu_x = X.mean(axis=0)
    std_x = X.std(axis=0)
    std_x[std_x < 1e-8] = 1.0
    X_std = (X - mu_x) / std_x
    
    mu_y = y_flat.mean()
    std_y = y_flat.std()
    if std_y < 1e-8:
        std_y = 1.0
    y_std = (y_flat - mu_y) / std_y

    # Rich polynomial feature function (degree 3)
    def poly_features(X, degree=3):
        cols = [np.ones(len(X))]
        for j in range(X.shape[1]):
            cols.append(X[:, j])
        for i in range(X.shape[1]):
            for j in range(i, X.shape[1]):
                cols.append(X[:, i] * X[:, j])
        if degree >= 3:
            for i in range(X.shape[1]):
                for j in range(i, X.shape[1]):
                    for k in range(j, X.shape[1]):
                        cols.append(X[:, i] * X[:, j] * X[:, k])
        return np.column_stack(cols)

    # Compute polynomial fallback as safety net
    X_poly = poly_features(X)
    w_poly, *_ = np.linalg.lstsq(X_poly, y_flat, rcond=None)
    
    def predict_poly(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        return poly_features(X_test) @ w_poly

    # Define Matern 5/2 kernel
    def matern_52_kernel(X1, X2, length_scale, variance):
        diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
        sq_dist = np.sum(diff**2, axis=-1)
        r = np.sqrt(sq_dist)
        sqrt5_r = np.sqrt(5.0) * r / length_scale
        K = variance * (1.0 + sqrt5_r + 5.0/3.0 * r**2 / (length_scale**2)) * np.exp(-sqrt5_r)
        return K

    # Log marginal likelihood
    def log_marginal_likelihood(params):
        length_scale = np.exp(params[0])
        variance = np.exp(params[1])
        noise = np.exp(params[2])
        
        K = matern_52_kernel(X_std, X_std, length_scale, variance)
        K += noise * np.eye(n)
        
        try:
            L, info = cho_factor(K, lower=True, check_finite=False)
            if info > 0:
                return -1e10
            alpha = cho_solve((L, info), y_std, check_finite=False)
            lml = 0.5 * y_std.dot(alpha) - 0.5 * np.sum(np.log(np.diag(L)**2)) - 0.5 * n * np.log(2.0 * np.pi)
            return -lml
        except:
            return -1e10

    # Multi-start hyperparameter optimization
    best_lml = -1e10
    best_params = None
    
    # Try multiple initializations
    initializations = [
        np.array([0.0, 0.0, -6.0]),
        np.array([-1.0, 0.5, -5.0]),
        np.array([0.5, 0.0, -7.0]),
        np.array([1.0, 1.0, -4.0]),
    ]
    
    for init in initializations:
        if time.time() > DEADLINE - 30:
            break
        try:
            result = minimize(log_marginal_likelihood, init, method='L-BFGS-B', 
                              options={'maxiter': 80, 'ftol': 1e-7, 'gtol': 1e-5})
            if result.fun < best_lml:
                best_lml = result.fun
                best_params = result.x.copy()
        except:
            pass

    # Check if GP is worth using
    use_gp = (best_params is not None and best_lml > -1e9)
    
    if not use_gp or time.time() > DEADLINE - 20:
        return predict_poly

    # Extract best hyperparameters
    best_length_scale = np.exp(best_params[0])
    best_variance = np.exp(best_params[1])
    best_noise = np.exp(best_params[2])

    # Precompute Cholesky factor for training data
    K_train = matern_52_kernel(X_std, X_std, best_length_scale, best_variance)
    K_train += best_noise * np.eye(n)
    
    try:
        L, info = cho_factor(K_train, lower=True, check_finite=False)
        if info > 0:
            raise ValueError("Cholesky factorization failed")
        alpha = cho_solve((L, info), y_std, check_finite=False)
    except:
        return predict_poly

    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        
        X_test_std = (X_test - mu_x) / std_x
        K_test_train = matern_52_kernel(X_test_std, X_std, best_length_scale, best_variance)
        y_pred_std = K_test_train.dot(alpha)
        y_pred = y_pred_std * std_y + mu_y
        y_pred = np.where(np.isfinite(y_pred), y_pred, mu_y)
        return y_pred

    return predict
# EVOLVE-BLOCK-END
