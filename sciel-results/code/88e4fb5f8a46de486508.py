"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist


def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 280.0  # safety margin before 300s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).flatten()
    n, d = X.shape
    
    # Normalize features for better kernel conditioning
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std < 1e-10] = 1.0
    X_norm = (X - X_mean) / X_std
    
    # Build multiple kernel models and ensemble them
    models = []
    
    # Model 1: RBF kernel with multiple bandwidths
    for gamma in [0.5, 1.0, 2.0, 5.0]:
        K = np.exp(-gamma * cdist(X_norm, X_norm) ** 2)
        alpha = np.linalg.solve(K + 1e-4 * np.eye(n), y)
        models.append(('rbf', gamma, X_mean, X_std, alpha))
    
    # Model 2: Polynomial kernel (degree 3)
    K_poly = (1 + X_norm @ X_norm.T / d) ** 3
    alpha_poly = np.linalg.solve(K_poly + 1e-4 * np.eye(n), y)
    models.append(('poly', 3, X_mean, X_std, alpha_poly))
    
    # Model 3: Linear kernel
    K_lin = X_norm @ X_norm.T
    alpha_lin = np.linalg.solve(K_lin + 1e-4 * np.eye(n), y)
    models.append(('linear', 1, X_mean, X_std, alpha_lin))
    
    # Model 4: RBF with adaptive bandwidth based on data spread
    dists = cdist(X_norm, X_norm)
    median_dist = np.median(dists[dists > 0]) if np.any(dists > 0) else 1.0
    gamma_adapt = 1.0 / max(median_dist ** 2, 1e-6)
    K_adapt = np.exp(-gamma_adapt * cdist(X_norm, X_norm) ** 2)
    alpha_adapt = np.linalg.solve(K_adapt + 1e-4 * np.eye(n), y)
    models.append(('rbf', gamma_adapt, X_mean, X_std, alpha_adapt))
    
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        X_test_norm = (X_test - X_mean) / X_std
        n_test = X_test.shape[0]
        
        preds = np.zeros(n_test)
        n_models = len(models)
        
        for mtype, gamma, mmean, mstd, alpha in models:
            if mtype == 'rbf':
                K_test = np.exp(-gamma * cdist(X_test_norm, X_norm) ** 2)
            elif mtype == 'poly':
                K_test = (1 + X_test_norm @ X_norm.T / d) ** 3
            else:  # linear
                K_test = X_test_norm @ X_norm.T
            preds += K_test @ alpha
        
        return preds / n_models
    
    return predict
# EVOLVE-BLOCK-END
