"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def fit(X_train, y_train):
    d = X_train.shape[1]
    n = len(X_train)
    
    # Normalize inputs to [0, 1] for better kernel conditioning
    X_min = X_train.min(axis=0, keepdims=True)
    X_max = X_train.max(axis=0, keepdims=True)
    X_range = X_max - X_min
    X_range[X_range < 1e-10] = 1.0
    
    def normalize(X):
        return (X - X_min) / X_range
    
    Xn_train = normalize(X_train)
    
    # Kernel Ridge Regression with RBF kernel
    # K(x, x') = exp(-gamma * ||x - x'||^2)
    # We search over gamma (inverse length scale squared) and alpha (regularization)
    
    # Precompute pairwise squared distances for training set
    # ||x - x'||^2 = ||x||^2 + ||x'||^2 - 2 x . x'
    Xn_sq = np.sum(Xn_train ** 2, axis=1, keepdims=True)
    dist_sq = Xn_sq + Xn_sq.T - 2 * (Xn_train @ Xn_train.T)
    # Clip small negative values due to floating point errors
    dist_sq = np.maximum(dist_sq, 0)
    
    # Hyperparameter search
    gammas = [1, 5, 10, 20, 50, 100]
    alphas = [1e-6, 1e-4, 1e-2, 1e-1, 1.0, 10.0]
    
    # 5-fold cross-validation
    n_folds = 5
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    best_gamma = 10.0
    best_alpha = 1e-4
    best_cv_mse = np.inf
    
    for gamma in gammas:
        # Compute full kernel matrix for this gamma
        K_full = np.exp(-gamma * dist_sq)
        
        for alpha in alphas:
            cv_mses = []
            for k in range(n_folds):
                val_idx = folds[k]
                train_idx = np.concatenate([folds[i] for i in range(n_folds) if i != k])
                
                # Extract sub-matrices
                K_train_val = K_full[np.ix_(train_idx, train_idx)]
                K_val_train = K_full[np.ix_(val_idx, train_idx)]
                
                y_tr = y_train[train_idx]
                y_val = y_train[val_idx]
                
                # Solve (K + alpha * I) w = y
                try:
                    w = np.linalg.solve(K_train_val + alpha * np.eye(len(train_idx)), y_tr)
                except np.linalg.LinAlgError:
                    w = np.linalg.lstsq(K_train_val + alpha * np.eye(len(train_idx)), y_tr, rcond=None)[0]
                
                pred = K_val_train @ w
                mse = np.mean((pred - y_val) ** 2)
                cv_mses.append(mse)
            
            mean_cv_mse = np.mean(cv_mses)
            if mean_cv_mse < best_cv_mse:
                best_cv_mse = mean_cv_mse
                best_gamma = gamma
                best_alpha = alpha
    
    # Final fit with best hyperparameters
    K_full = np.exp(-best_gamma * dist_sq)
    try:
        w_final = np.linalg.solve(K_full + best_alpha * np.eye(n), y_train)
    except np.linalg.LinAlgError:
        w_final = np.linalg.lstsq(K_full + best_alpha * np.eye(n), y_train, rcond=None)[0]
    
    def predict(X):
        Xn = normalize(X)
        # Compute kernel between test points and training points
        # K_test_train (n_test x n_train)
        # dist_sq_test_train[i, j] = ||Xn[i] - Xn_train[j]||^2
        Xn_sq_test = np.sum(Xn ** 2, axis=1, keepdims=True)
        dist_sq_test_train = Xn_sq_test + Xn_sq.T - 2 * (Xn @ Xn_train.T)
        dist_sq_test_train = np.maximum(dist_sq_test_train, 0)
        
        K_test_train = np.exp(-best_gamma * dist_sq_test_train)
        
        return K_test_train @ w_final
    
    return predict
# EVOLVE-BLOCK-END
