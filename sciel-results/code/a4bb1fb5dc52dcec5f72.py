"""
Strategy: Gaussian Process Regression with Matern 5/2 kernel.
This approach is robust for smooth, non-polynomial functions with limited data.
It optimizes length scale and signal variance to balance bias and variance.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


def fit(X_train, y_train):
    start_time = time.time()
    # Allocate ~70s per equation to stay well under the 330s total limit for 4 equations
    deadline = start_time + 70.0

    X = np.array(X_train).flatten()
    y = np.array(y_train).flatten()
    n = len(X)

    # Standardize X for better numerical stability and hyperparameter optimization
    mu_x = np.mean(X)
    sigma_x = np.std(X)
    if sigma_x < 1e-8:
        sigma_x = 1.0
    X_std = (X - mu_x) / sigma_x

    # Matern 5/2 kernel
    def matern52_kernel(X1, X2, l, sigma):
        # X1, X2 are (N, 1)
        d = cdist(X1, X2, metric='euclidean') / l
        K = sigma**2 * (1 + np.sqrt(5)*d + (5/3)*d**2) * np.exp(-np.sqrt(5)*d)
        return K

    def neg_log_likelihood(params):
        if time.time() > deadline:
            return 1e10
        log_l, log_sigma = params
        l = np.exp(log_l)
        sigma = np.exp(log_sigma)
        
        # Small noise floor for numerical stability
        noise = 1e-6
        K = matern52_kernel(X_std.reshape(-1, 1), X_std.reshape(-1, 1), l, sigma)
        K += noise * np.eye(n)
        
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return 1e10
            
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        log_det = 2 * np.sum(np.log(np.diag(L)))
        nll = 0.5 * np.dot(y, alpha) + 0.5 * log_det + 0.5 * n * np.log(2 * np.pi)
        
        return nll

    # Estimate initial sigma from data
    sigma_y = np.std(y)
    if sigma_y < 1e-8:
        sigma_y = 1.0

    best_nll = np.inf
    best_params = None

    # Multiple initializations to avoid local minima
    inits = [
        [np.log(0.1), np.log(sigma_y)],
        [np.log(0.5), np.log(sigma_y)],
        [np.log(1.0), np.log(sigma_y)],
        [np.log(2.0), np.log(sigma_y)],
        [np.log(0.01), np.log(sigma_y)],
    ]

    for init in inits:
        if time.time() > deadline - 5:
            break
        try:
            res = minimize(neg_log_likelihood, init, method='L-BFGS-B', 
                           bounds=[[-6, 6], [-6, 6]], options={'maxiter': 100})
            if res.fun < best_nll:
                best_nll = res.fun
                best_params = res.x
        except Exception:
            continue

    if best_params is None:
        # Fallback if optimization fails completely
        best_params = [np.log(0.5), np.log(sigma_y)]
        
    log_l, log_sigma = best_params
    l = np.exp(log_l)
    sigma = np.exp(log_sigma)
    noise = 1e-6
    
    # Pre-compute Cholesky decomposition for prediction
    K_train = matern52_kernel(X_std.reshape(-1, 1), X_std.reshape(-1, 1), l, sigma)
    K_train += noise * np.eye(n)
    
    try:
        L = np.linalg.cholesky(K_train)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    except np.linalg.LinAlgError:
        # Fallback to degree-3 polynomial if GP fails
        def features_poly(X):
            X = np.array(X).flatten()
            return np.column_stack([np.ones_like(X), X, X**2, X**3])
        w, *_ = np.linalg.lstsq(features_poly(X), y, rcond=None)
        def predict_poly(X_new):
            return features_poly(X_new) @ w
        return predict_poly

    def predict(X_new):
        X_new = np.array(X_new).flatten()
        X_new_std = (X_new - mu_x) / sigma_x
        K_new = matern52_kernel(X_new_std.reshape(-1, 1), X_std.reshape(-1, 1), l, sigma)
        pred = K_new @ alpha
        return pred

    return predict
# EVOLVE-BLOCK-END
