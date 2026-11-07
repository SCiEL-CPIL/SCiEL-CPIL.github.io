import numpy as np
from scipy.linalg import cho_factor, cho_solve

def fit(X_train, y_train):
    # X_train is (N, 1)
    x = X_train[:, 0]
    y = y_train
    
    # Normalize x to [-1, 1]
    x_min, x_max = x.min(), x.max()
    if x_max == x_min:
        x_norm = np.zeros_like(x)
    else:
        x_norm = 2 * (x - x_min) / (x_max - x_min) - 1
        
    # Kernel function
    def matern52(X1, X2, length_scale):
        # X1, X2 are 1D arrays
        r = np.abs(X1[:, None] - X2[None, :])
        sqrt5 = np.sqrt(5)
        z = sqrt5 * r / length_scale
        K = (1 + z + z**2/3) * np.exp(-z)
        return K

    # Grid search for hyperparameters
    best_lml = -np.inf
    best_params = None
    
    # Log-spaced grids
    length_scales = np.logspace(-1, 1, 10) # 0.1 to 10
    noises = np.logspace(-8, -2, 10) # 1e-8 to 1e-2
    
    # Signal variance
    sigma_f2 = np.var(y)
    if sigma_f2 == 0:
        sigma_f2 = 1.0

    for ls in length_scales:
        K = matern52(x_norm, x_norm, ls)
        for noise in noises:
            K_noise = K * sigma_f2 + noise * np.eye(len(x))
            try:
                # Compute LML
                L = np.linalg.cholesky(K_noise)
                # alpha = solve(L, solve(L.T, y))
                # LML = 0.5 * y.T @ alpha - sum(log(diag(L))) - N/2 * log(2*pi)
                y_L = np.linalg.solve(L, y)
                lml = -0.5 * np.dot(y, y_L) - np.sum(np.log(np.diag(L))) - 0.5 * len(x) * np.log(2 * np.pi)
                if lml > best_lml:
                    best_lml = lml
                    best_params = (ls, noise, L, y_L)
            except:
                continue
                
    if best_params is None:
        # Fallback to mean
        return lambda X: np.full(X.shape[0], np.mean(y))
        
    ls, noise, L, y_L = best_params
    
    def predict(X):
        x_test = X[:, 0]
        x_test_norm = 2 * (x_test - x_min) / (x_max - x_min) - 1
        K_star = matern52(x_test_norm, x_norm, ls) * sigma_f2
        # prediction mean = K_star @ solve(K_noise, y) = K_star @ y_L
        return K_star @ y_L

    return predict
