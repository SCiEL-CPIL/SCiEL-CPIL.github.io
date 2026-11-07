import numpy as np
from scipy.spatial.distance import cdist

def fit(X_train, y_train):
    X = X_train
    y = y_train
    
    # Normalize X for better kernel conditioning
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std

    def rbf_kernel(X1, X2, gamma):
        sq_norms1 = np.sum(X1**2, axis=1)
        sq_norms2 = np.sum(X2**2, axis=1)
        dist_sq = sq_norms1[:, None] + sq_norms2[None, :] - 2 * X1 @ X2.T
        dist_sq = np.maximum(dist_sq, 0)
        return np.exp(-gamma * dist_sq)

    def poly_kernel(X1, X2, degree, coef0=1.0):
        return (X1 @ X2.T + coef0) ** degree

    # Hyperparameter grid
    # Include both RBF and Polynomial kernels
    # We'll use a tuple (type, param1, param2) to represent kernel configs
    # type 0: RBF, param1: gamma
    # type 1: Poly, param1: degree, param2: coef0
    
    kernel_configs = []
    # RBF gammas
    for gamma in [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]:
        kernel_configs.append(('rbf', gamma, 0))
    
    # Polynomial degrees
    for degree in [2, 4, 6, 8, 10, 12, 15, 20]:
        for coef0 in [0.0, 1.0, 2.0]:
            kernel_configs.append(('poly', degree, coef0))
            
    alphas = [1e-8, 1e-6, 1e-4, 1e-2, 1.0]
    
    # Cross-validation to select best kernel and alpha
    n = len(X_norm)
    k_folds = 5
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k_folds)
    
    best_score = np.inf
    best_config = ('rbf', 1.0, 0)
    best_alpha = 1e-2
    
    for k_type, p1, p2 in kernel_configs:
        for alpha in alphas:
            cv_errors = []
            valid = True
            for i in range(k_folds):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(k_folds) if j != i])
                
                X_tr = X_norm[train_idx]
                y_tr = y[train_idx]
                X_val = X_norm[val_idx]
                y_val = y[val_idx]
                
                if k_type == 'rbf':
                    K_train = rbf_kernel(X_tr, X_tr, p1)
                    K_val = rbf_kernel(X_val, X_tr, p1)
                else:
                    K_train = poly_kernel(X_tr, X_tr, p1, p2)
                    K_val = poly_kernel(X_val, X_tr, p1, p2)
                
                # Solve (K + alpha*I) w = y
                try:
                    A = K_train + alpha * np.eye(len(X_tr))
                    # Use Cholesky for SPD matrices
                    L = np.linalg.cholesky(A)
                    w = np.linalg.solve(L.T, np.linalg.solve(L, y_tr))
                    pred_val = K_val @ w
                    mse = np.mean((pred_val - y_val)**2)
                    cv_errors.append(mse)
                except (np.linalg.LinAlgError, ValueError):
                    valid = False
                    break
            
            if valid:
                mean_cv_error = np.mean(cv_errors)
                if mean_cv_error < best_score:
                    best_score = mean_cv_error
                    best_config = (k_type, p1, p2)
                    best_alpha = alpha

    # Final fit on all data
    k_type, p1, p2 = best_config
    if k_type == 'rbf':
        K_train = rbf_kernel(X_norm, X_norm, p1)
    else:
        K_train = poly_kernel(X_norm, X_norm, p1, p2)
        
    try:
        A = K_train + best_alpha * np.eye(len(X_norm))
        L = np.linalg.cholesky(A)
        w = np.linalg.solve(L.T, np.linalg.solve(L, y))
    except (np.linalg.LinAlgError, ValueError):
        # Fallback to least squares if singular
        A = K_train + best_alpha * np.eye(len(X_norm))
        w, *_ = np.linalg.lstsq(A, y, rcond=None)

    def predict(X):
        X_test = (X - X_mean) / X_std
        if best_config[0] == 'rbf':
            K_test = rbf_kernel(X_test, X_norm, best_config[1])
        else:
            K_test = poly_kernel(X_test, X_norm, best_config[1], best_config[2])
        return K_test @ w

    return predict
