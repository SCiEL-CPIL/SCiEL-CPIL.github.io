"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import solve

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 250.0  # 300s limit, reserve 50s for safety
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64).flatten()
    n, d = X.shape
    
    # Normalize inputs to [0, 1] for better kernel behavior
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range < 1e-10] = 1.0
    X_norm = (X - X_min) / X_range
    
    def predict_norm(X_pred):
        return (X_pred - X_min) / X_range
    
    def rbf_kernel(X1, X2, gamma):
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
        sq_dists = np.maximum(sq_dists, 0)
        return np.exp(-gamma * sq_dists)
    
    def poly_kernel(X1, X2, degree=3, coef0=1.0):
        return (X1 @ X2.T + coef0) ** degree
    
    def linear_kernel(X1, X2):
        return X1 @ X2.T
    
    def fit_krr(X_train_norm, y_train, kernel_fn, alpha, lam):
        """Fit Kernel Ridge Regressor. Returns predict function for normalized inputs."""
        K = kernel_fn(X_train_norm, X_train_norm)
        n = len(y_train)
        # Solve (K + lam*I + alpha) * w = y
        A = K + (lam + alpha) * np.eye(n)
        try:
            w = solve(A, y_train, assume_a='pos')
        except:
            w = np.linalg.lstsq(A, y_train, rcond=None)[0]
        
        def predict(X_pred_norm):
            K_test = kernel_fn(X_pred_norm, X_train_norm)
            return K_test @ w
        return predict
    
    # Generate candidate models
    candidates = []
    
    # RBF kernels with different gammas
    gamma_values = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    for gamma in gamma_values:
        candidates.append(('rbf', gamma, 1e-3))
        candidates.append(('rbf', gamma, 1e-2))
    
    # Polynomial kernels
    for degree in [2, 3, 4]:
        for coef0 in [0.5, 1.0, 2.0]:
            candidates.append(('poly', (degree, coef0), 1e-3))
    
    # Linear kernel
    candidates.append(('linear', 1.0, 1e-3))
    
    def get_kernel_fn(model_type, param, lam):
        if model_type == 'rbf':
            return lambda X1, X2: rbf_kernel(X1, X2, param)
        elif model_type == 'poly':
            degree, coef0 = param
            return lambda X1, X2: poly_kernel(X1, X2, degree, coef0)
        elif model_type == 'linear':
            return lambda X1, X2: linear_kernel(X1, X2)
    
    # Cross-validation to select best model
    n_folds = 5
    fold_indices = np.array_split(np.arange(n), n_folds)
    
    best_score = np.inf
    best_predict = None
    best_model_info = None
    
    for model_type, param, lam in candidates:
        if time.time() > deadline - 30:  # Need time to fit final model
            break
        
        cv_errors = []
        for i in range(n_folds):
            val_idx = fold_indices[i]
            train_idx = np.concatenate([fold_indices[j] for j in range(n_folds) if j != i])
            
            X_tr = X_norm[train_idx]
            y_tr = y[train_idx]
            X_val = X_norm[val_idx]
            y_val = y[val_idx]
            
            kernel_fn = get_kernel_fn(model_type, param, lam)
            pred_fn = fit_krr(X_tr, y_tr, kernel_fn, 0.0, lam)
            preds = pred_fn(X_val)
            mse = np.mean((preds - y_val) ** 2)
            cv_errors.append(mse)
        
        mean_cv_error = np.mean(cv_errors)
        
        if mean_cv_error < best_score:
            best_score = mean_cv_error
            best_model_info = (model_type, param, lam)
        
        # Early stopping if we have a very good model
        if best_score < 1e-6 and best_model_info is not None:
            break
    
    # Fit final model with best parameters on all data
    if best_model_info is not None:
        model_type, param, lam = best_model_info
        kernel_fn = get_kernel_fn(model_type, param, lam)
        final_predict_norm = fit_krr(X_norm, y, kernel_fn, 0.0, lam)
    else:
        # Fallback: simple linear regression
        from numpy.linalg import lstsq
        X_aug = np.column_stack([np.ones(n), X_norm])
        w, *_ = lstsq(X_aug, y, rcond=None)
        def final_predict_norm(X_pred_norm):
            X_aug_pred = np.column_stack([np.ones(len(X_pred_norm)), X_pred_norm])
            return X_aug_pred @ w
    
    def predict(X):
        X_pred_norm = predict_norm(np.asarray(X, dtype=np.float64))
        return final_predict_norm(X_pred_norm)
    
    return predict
# EVOLVE-BLOCK-END
