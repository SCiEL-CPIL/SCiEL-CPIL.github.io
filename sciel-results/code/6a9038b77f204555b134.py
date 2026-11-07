"""
Strategy: Kernel Ridge Regression with Gaussian (RBF) Kernel.
We perform a grid search on the kernel bandwidth (sigma) and regularization
parameter (lambda) using 5-fold cross-validation to adapt to the specific
hidden equation. This approach is significantly more flexible than a fixed
degree polynomial and can approximate a wide variety of functional forms
(trig, exp, polynomial, etc.).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    
    # Standardize X for better kernel behavior
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    X_std[X_std == 0] = 1.0 # Avoid division by zero
    
    X_train_std = (X_train - X_mean) / X_std
    y_train = np.asarray(y_train)
    
    n, d = X_train_std.shape
    
    # Grid search parameters
    sigmas = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    lambdas = [1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0, 100.0]
    
    best_score = np.inf
    best_sigma = 1.0
    best_lambda = 1.0
    
    # 5-Fold Cross Validation
    k = 5
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, k)
    
    for sigma in sigmas:
        # Pre-compute kernel matrix for this sigma if needed? 
        # No, because we need train/test splits.
        for lam in lambdas:
            cv_errors = []
            for i in range(k):
                test_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(k) if j != i])
                
                X_tr = X_train_std[train_idx]
                y_tr = y_train[train_idx]
                X_te = X_train_std[test_idx]
                y_te = y_train[test_idx]
                
                # Compute kernel matrices
                # K_train: (n_tr, n_tr)
                # K_test_train: (n_te, n_tr)
                
                def rbf_kernel(A, B, sigma):
                    # A: (n1, d), B: (n2, d)
                    # dist^2 = ||A||^2 + ||B||^2 - 2 A B^T
                    A_sq = np.sum(A**2, axis=1)
                    B_sq = np.sum(B**2, axis=1)
                    dist_sq = A_sq[:, None] + B_sq[None, :] - 2 * (A @ B.T)
                    # Clip negative values due to numerical error
                    dist_sq = np.maximum(dist_sq, 0)
                    return np.exp(-dist_sq / (2 * sigma**2))
                
                K_tr = rbf_kernel(X_tr, X_tr, sigma)
                K_te_tr = rbf_kernel(X_te, X_tr, sigma)
                
                # Solve (K_tr + lam * I) w = y_tr
                n_tr = len(y_tr)
                try:
                    w = np.linalg.solve(K_tr + lam * np.eye(n_tr), y_tr)
                    y_pred = K_te_tr @ w
                    mse = np.mean((y_pred - y_te)**2)
                    cv_errors.append(mse)
                except np.linalg.LinAlgError:
                    cv_errors.append(np.inf)
            
            mean_cv_error = np.mean(cv_errors)
            if mean_cv_error < best_score:
                best_score = mean_cv_error
                best_sigma = sigma
                best_lambda = lam
                
                # Time check: If we are taking too long, break early
                if time.time() - start_time > 250:
                    break
            
            if time.time() - start_time > 250:
                break
        
        if time.time() - start_time > 250:
            break
    
    # Final fit with best parameters on full training set
    K_train = rbf_kernel(X_train_std, X_train_std, best_sigma)
    n_train = len(y_train)
    try:
        w_final = np.linalg.solve(K_train + best_lambda * np.eye(n_train), y_train)
    except np.linalg.LinAlgError:
        # Fallback to lstsq if singular
        w_final, *_ = np.linalg.lstsq(K_train + best_lambda * np.eye(n_train), y_train, rcond=None)
    
    def predict(X):
        X_std = (X - X_mean) / X_std
        K_test_train = rbf_kernel(X_std, X_train_std, best_sigma)
        return K_test_train @ w_final

    return predict
# EVOLVE-BLOCK-END
