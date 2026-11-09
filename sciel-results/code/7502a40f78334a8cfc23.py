import numpy as np
from scipy.optimize import minimize
import time

def fit(X_train, y_train):
    # Record start time for deadline guard
    start_time = time.time()
    deadline = 300.0  # seconds
    safety_margin = 10.0

    X = np.asarray(X_train, dtype=np.float64).ravel()
    y = np.asarray(y_train, dtype=np.float64).ravel()
    n = len(X)

    # Standardize X for better conditioning
    x_mean = np.mean(X)
    x_std = np.std(X)
    if x_std < 1e-12:
        x_std = 1.0
    X_scaled = (X - x_mean) / x_std

    # GP Kernel: Matern 5/2
    def matern52_kernel(x1, x2, theta):
        log_l, log_sf = theta
        l = np.exp(log_l)
        sf = np.exp(log_sf)
        # Distance matrix
        diff = x1[:, np.newaxis] - x2[np.newaxis, :]
        r = np.sqrt(np.sum(diff**2, axis=-1))
        # Matern 5/2 formula
        kr = r / l
        term = 1.0 + kr + (kr**2) / 3.0
        k = (sf**2) * term * np.exp(-kr)
        return k

    def log_marginal_likelihood(theta):
        K = matern52_kernel(X_scaled, X_scaled, theta)
        # Add small noise for numerical stability
        K += 1e-6 * np.eye(n)
        
        try:
            # Cholesky decomposition for stability
            L = np.linalg.cholesky(K)
            # Solve L alpha = y
            alpha = np.linalg.solve(L, y)
            # Solve L^T v = alpha
            v = np.linalg.solve(L.T, alpha)
            # Log det = 2 * sum(log(diag(L)))
            log_det = 2.0 * np.sum(np.log(np.diag(L)))
            # Compute LML
            lml = -0.5 * (y @ alpha + log_det + n * np.log(2.0 * np.pi))
            return -lml  # Minimize negative LML
        except np.linalg.LinAlgError:
            return 1e10

    # Initial guess: length scale ~ 1.0, signal variance ~ std(y)
    y_std = np.std(y)
    if y_std < 1e-12:
        y_std = 1.0
    theta0 = np.array([0.0, np.log(y_std)])  # log(1.0), log(y_std)
    
    bounds = [(-5.0, 5.0), (-5.0, 5.0)]  # log scale bounds

    # Optimize hyperparameters
    best_theta = theta0
    best_score = np.inf
    
    # Use L-BFGS-B
    try:
        res = minimize(log_marginal_likelihood, theta0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 100})
        if res.fun < best_score:
            best_score = res.fun
            best_theta = res.x
    except Exception:
        pass

    # Check deadline
    if time.time() > deadline - safety_margin:
        # If we are running out of time, just use the initial guess or current best
        pass

    # Compute inverse and alpha for prediction
    K = matern52_kernel(X_scaled, X_scaled, best_theta)
    K += 1e-6 * np.eye(n)
    
    try:
        L = np.linalg.cholesky(K)
        alpha = np.linalg.solve(L, y)
        v = np.linalg.solve(L.T, alpha)
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse if Cholesky fails
        try:
            K_inv = np.linalg.pinv(K)
            alpha = K_inv @ y
        except Exception:
            # Ultimate fallback: constant prediction
            mean_y = np.mean(y)
            return lambda X_test: np.full(X_test.shape[0], mean_y)

    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64).ravel()
        X_test_scaled = (X_test - x_mean) / x_std
        
        # Kernel between test and train
        K_star = matern52_kernel(X_test_scaled, X_scaled, best_theta)
        
        # Prediction mean
        # f_* = K_star @ alpha
        f_pred = K_star @ alpha
        
        # Ensure finite values
        f_pred = np.nan_to_num(f_pred, nan=0.0, posinf=0.0, neginf=0.0)
        
        return f_pred

    return predict
