import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280.0  # Leave 50s buffer for safety

    X = np.array(X_train, dtype=float)
    y = np.array(y_train, dtype=float)
    n, d = X.shape

    # Handle edge case: constant y
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-10:
        def predict_const(X_test):
            X_test = np.array(X_test, dtype=float)
            if X_test.ndim == 1:
                return np.array([y_mean])
            return np.full(X_test.shape[0], y_mean)
        return predict_const

    # Normalize X
    x_mean = X.mean(axis=0)
    x_std = X.std(axis=0)
    x_std[x_std < 1e-10] = 1.0
    X_norm = (X - x_mean) / x_std

    # Normalize y
    y_norm = (y - y_mean) / y_std

    def _time_left():
        return deadline - time.time()

    # Matern 5/2 kernel with ARD
    def matern52_kernel(X1, X2, lengthscales):
        """Compute Matern 5/2 kernel matrix between X1 and X2."""
        # X1: (n1, d), X2: (n2, d), lengthscales: (d,)
        # Distance matrix with ARD scaling
        # D_ij^2 = sum_k (X1_ik - X2_jk)^2 / l_k^2
        X1_scaled = X1 / lengthscales
        X2_scaled = X2 / lengthscales
        sq_dists = cdist(X1_scaled, X2_scaled, 'sqeuclidean')
        sq_dists = np.maximum(sq_dists, 0.0)
        # Matern 5/2: (1 + sqrt(3)*r + 3*r^2/2) * exp(-sqrt(3)*r)
        r = np.sqrt(sq_dists)
        sqrt3_r = np.sqrt(3.0) * r
        return (1.0 + sqrt3_r + 1.5 * sq_dists) * np.exp(-sqrt3_r)

    def neg_log_marginal_likelihood(theta):
        """Negative log marginal likelihood for GP."""
        if _time_left() < 5:
            return 1e10
        
        # theta = [log(l_1), ..., log(l_d), log(sigma_f), log(sigma_n)]
        log_ls = theta[:d]
        log_sigma_f = theta[d]
        log_sigma_n = theta[d+1]
        
        lengthscales = np.exp(log_ls)
        sigma_f = np.exp(log_sigma_f)
        sigma_n = np.exp(log_sigma_n)
        
        # Compute kernel matrix
        K = matern52_kernel(X_norm, X_norm, lengthscales)
        K = sigma_f**2 * K + sigma_n**2 * np.eye(n)
        
        # Ensure positive definite
        try:
            L = np.linalg.cholesky(K + 1e-10 * np.eye(n))
        except:
            return 1e10
        
        # Log marginal likelihood:
        # log p(y|X) = -0.5 * y^T K^{-1} y - 0.5 * log det(K) - n/2 * log(2*pi)
        # log det(K) = 2 * sum(log(diag(L)))
        # K^{-1} y = solve(L.T, solve(L, y))
        try:
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            nll = 0.5 * y_norm @ alpha + 0.5 * log_det + 0.5 * n * np.log(2.0 * np.pi)
        except:
            return 1e10
        
        return nll

    # Initial hyperparameters
    # Length scales: try values in [0.1, 10] in normalized space
    # sigma_f: around 1 (since y is normalized)
    # sigma_n: small noise, around 0.01-0.1
    
    best_nll = np.inf
    best_theta = None
    
    # Multi-start optimization
    n_starts = 8
    np.random.seed(42)
    
    for i in range(n_starts):
        if _time_left() < 30:
            break
        
        # Random initial values
        if i == 0:
            # Start with reasonable defaults
            log_ls_init = np.zeros(d)  # lengthscales = 1
            log_sf_init = 0.0  # sigma_f = 1
            log_sn_init = -2.0  # sigma_n = 0.01
        else:
            log_ls_init = np.random.uniform(-2, 2, d)
            log_sf_init = np.random.uniform(-1, 1)
            log_sn_init = np.random.uniform(-3, 0)
        
        theta0 = np.concatenate([log_ls_init, [log_sf_init, log_sn_init]])
        
        # Bounds
        bounds = [(-3, 3)] * d + [(-2, 2), (-4, 1)]
        
        try:
            result = minimize(neg_log_marginal_likelihood, theta0, method='L-BFGS-B',
                             bounds=bounds, options={'maxiter': 200, 'ftol': 1e-8})
            
            if result.fun < best_nll:
                best_nll = result.fun
                best_theta = result.x
        except:
            continue
    
    # Fallback: if optimization failed, use reasonable defaults
    if best_theta is None:
        best_theta = np.concatenate([np.zeros(d), [0.0, -2.0]])
    
    # Extract final hyperparameters
    log_ls = best_theta[:d]
    log_sigma_f = best_theta[d]
    log_sigma_n = best_theta[d+1]
    
    lengthscales = np.exp(log_ls)
    sigma_f = np.exp(log_sigma_f)
    sigma_n = np.exp(log_sigma_n)
    
    # Compute final kernel matrix and solve for weights
    K = matern52_kernel(X_norm, X_norm, lengthscales)
    K = sigma_f**2 * K + sigma_n**2 * np.eye(n)
    
    try:
        L = np.linalg.cholesky(K + 1e-10 * np.eye(n))
        # Weights for prediction: w = K^{-1} y
        # K = L L^T, so K^{-1} = L^{-T} L^{-1}
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
    except:
        # Fallback to least squares
        try:
            alpha = np.linalg.solve(K, y_norm)
        except:
            alpha = np.zeros(n)
    
    def predict(X_test):
        X_test = np.array(X_test, dtype=float)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        
        X_test_norm = (X_test - x_mean) / x_std
        
        # Compute kernel between test and train
        K_test = matern52_kernel(X_test_norm, X_norm, lengthscales)
        K_test = sigma_f**2 * K_test
        
        y_pred_norm = K_test @ alpha
        y_pred = y_pred_norm * y_std + y_mean
        
        return y_pred
    
    return predict
