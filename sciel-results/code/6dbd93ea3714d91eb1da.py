"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin for 330s limit
    
    # Flatten X for 1D processing
    X = np.asarray(X_train, dtype=float).reshape(-1, 1)
    y = np.asarray(y_train, dtype=float).flatten()
    n = len(y)
    
    # Kernel function: Matérn 5/2
    def matern52_kernel(X1, X2, length_scale, variance):
        d = cdist(X1, X2, 'euclidean')
        sqrt3d = np.sqrt(3) * d / length_scale
        K = variance * (1 + sqrt3d + (sqrt3d ** 2) / 3) * np.exp(-sqrt3d)
        return K

    # Log marginal likelihood (minimize negative)
    def neg_log_marginal_likelihood(params):
        if time.time() > deadline:
            return 1e10
        length_scale, variance, noise = params
        if length_scale <= 0 or variance <= 0 or noise <= 0:
            return 1e10
        
        K = matern52_kernel(X, X, length_scale, variance)
        K += noise * np.eye(n)
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            log_det = 2 * np.sum(np.log(np.diag(L)))
            lml = -0.5 * y.dot(alpha) - 0.5 * log_det - 0.5 * n * np.log(2 * np.pi)
            return -lml
        except np.linalg.LinAlgError:
            return 1e10

    # Use log-transformed parameters for L-BFGS-B (unconstrained optimization)
    def neg_lml_log_params(log_params):
        if time.time() > deadline:
            return 1e10
        length_scale = np.exp(log_params[0])
        variance = np.exp(log_params[1])
        noise = np.exp(log_params[2])
        return neg_log_marginal_likelihood(np.array([length_scale, variance, noise]))

    # Estimate initial parameters
    x_range = np.max(X) - np.min(X)
    if x_range <= 0:
        x_range = 1.0
    y_var = np.var(y)
    if y_var < 1e-10:
        y_var = 1.0
    
    # Multiple starting points for length scale
    ls_starts = [x_range * f for f in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]]
    var_starts = [y_var * f for f in [0.1, 1.0, 10.0]]
    noise_starts = [1e-8, 1e-6, 1e-4]
    
    best_params = None
    best_nll = 1e10
    
    # Multi-start with L-BFGS-B
    for ls in ls_starts:
        for var in var_starts:
            for noise in noise_starts:
                if time.time() > deadline:
                    break
                x0_log = np.array([np.log(ls), np.log(var), np.log(noise)])
                try:
                    res = minimize(neg_lml_log_params, x0_log, method='L-BFGS-B',
                                   options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
                    if res.fun < best_nll:
                        best_nll = res.fun
                        best_params = res.x
                except Exception:
                    pass
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break
    
    # If all optimizations failed, use reasonable defaults
    if best_params is None or best_nll >= 1e9:
        length_scale = x_range / 5.0
        variance = y_var
        noise = 1e-6
    else:
        length_scale = np.exp(best_params[0])
        variance = np.exp(best_params[1])
        noise = np.exp(best_params[2])
    
    # Ensure valid parameters
    if length_scale <= 0:
        length_scale = x_range / 5.0
    if variance <= 0:
        variance = y_var
    if noise <= 0:
        noise = 1e-6
    
    # Precompute for prediction
    K_train = matern52_kernel(X, X, length_scale, variance)
    K_train += noise * np.eye(n)
    
    try:
        L = np.linalg.cholesky(K_train)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
    except np.linalg.LinAlgError:
        # Fallback: polynomial regression
        def poly_features(X_in):
            X_flat = X_in.reshape(-1)
            cols = [np.ones(len(X_flat))]
            for p in range(1, 11):
                cols.append(X_flat ** p)
            return np.column_stack(cols)
        w, *_ = np.linalg.lstsq(poly_features(X_train), y_train, rcond=None)
        return lambda X: poly_features(X) @ w

    def predict(X_pred):
        X_pred = np.asarray(X_pred, dtype=float).reshape(-1, 1)
        K_star = matern52_kernel(X_pred, X, length_scale, variance)
        mu = K_star.dot(alpha)
        # Ensure finite output
        mu = np.where(np.isfinite(mu), mu, 0.0)
        return mu

    return predict
# EVOLVE-BLOCK-END
