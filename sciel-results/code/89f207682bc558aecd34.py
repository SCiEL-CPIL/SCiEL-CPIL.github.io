"""Symbolic regression using Gaussian Process Regression with hyperparameter optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.linalg import cho_factor, cho_solve
import warnings

def fit(X_train, y_train):
    t_start = time.time()
    DEADLINE = 320.0  # Safety margin before 330s kill

    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    # Handle potential edge cases in data
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

    # Define the Matern 5/2 kernel
    def matern_52_kernel(X1, X2, length_scale, variance):
        # X1: (n1, d), X2: (n2, d)
        # Calculate squared distances
        # To avoid memory issues for large n, we assume n is small (200)
        diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
        sq_dist = np.sum(diff**2, axis=-1)
        r = np.sqrt(sq_dist)
        # Matern 5/2
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

    # Initial hyperparameters (log space)
    # length_scale ~ 1.0, variance ~ 1.0, noise ~ 1e-3
    initial_params = np.array([0.0, 0.0, -6.0])

    best_lml = -1e10
    best_params = initial_params
    
    # Optimization
    try:
        result = minimize(log_marginal_likelihood, initial_params, method='L-BFGS-B', 
                          options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-4})
        
        if result.fun < best_lml:
            best_lml = result.fun
            best_params = result.x
    except Exception as e:
        pass

    # Check time
    if time.time() > DEADLINE:
        # Fallback to simple polynomial if too slow
        def features(X):
            cols = [np.ones(len(X))] + [X[:, j] for j in range(X.shape[1])] + [X[:, i] * X[:, j] for i in range(X.shape[1]) for j in range(i, X.shape[1])]
            return np.column_stack(cols)
        w, *_ = np.linalg.lstsq(features(X_train), y_train, rcond=None)
        return lambda X: features(X) @ w

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
        # Fallback if GP fails
        def features(X):
            cols = [np.ones(len(X))] + [X[:, j] for j in range(X.shape[1])] + [X[:, i] * X[:, j] for i in range(X.shape[1]) for j in range(i, X.shape[1])]
            return np.column_stack(cols)
        w, *_ = np.linalg.lstsq(features(X_train), y_train, rcond=None)
        return lambda X: features(X) @ w

    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        if X_test.ndim == 1:
            X_test = X_test.reshape(-1, 1)
        
        # Standardize test data
        X_test_std = (X_test - mu_x) / std_x
        
        # Compute kernel between test and train
        K_test_train = matern_52_kernel(X_test_std, X_std, best_length_scale, best_variance)
        
        # Predict
        y_pred_std = K_test_train.dot(alpha)
        
        # Inverse standardize
        y_pred = y_pred_std * std_y + mu_y
        
        # Ensure finite
        y_pred = np.where(np.isfinite(y_pred), y_pred, mu_y)
        
        return y_pred

    return predict
# EVOLVE-BLOCK-END
