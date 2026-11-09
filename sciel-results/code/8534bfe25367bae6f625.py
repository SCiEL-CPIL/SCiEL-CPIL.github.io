"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import combinations_with_replacement


def fit(X_train, y_train):
    t_start = time.time()
    deadline = 290.0  # safety margin before 300s limit
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    # Normalize inputs for better kernel conditioning
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X_norm = (X - X_mean) / X_std
    
    def rbf_kernel(X1, X2, length_scales):
        """RBF kernel with per-dimension length scales."""
        # X1: (n1, d), X2: (n2, d)
        # Compute squared distances
        sq1 = np.sum(X1**2, axis=1).reshape(-1, 1)
        sq2 = np.sum(X2**2, axis=1).reshape(1, -1)
        dist_sq = sq1 + sq2 - 2 * X1 @ X2.T
        # Per-dimension scaling
        # dist_sq is already sum over dims of (x1_j - x2_j)^2
        # We want sum_j (x1_j - x2_j)^2 / l_j^2
        # This equals sum_j (x1_j^2/l_j^2 + x2_j^2/l_j^2 - 2*x1_j*x2_j/l_j^2)
        # So we need to recompute with scaling
        X1_scaled = X1 / length_scales
        X2_scaled = X2 / length_scales
        sq1_s = np.sum(X1_scaled**2, axis=1).reshape(-1, 1)
        sq2_s = np.sum(X2_scaled**2, axis=1).reshape(1, -1)
        dist_sq_s = sq1_s + sq2_s - 2 * X1_scaled @ X2_scaled.T
        dist_sq_s = np.maximum(dist_sq_s, 0)
        return np.exp(-0.5 * dist_sq_s)
    
    def poly_kernel(X1, X2, degree, coef0=1.0):
        """Polynomial kernel."""
        return (coef0 + X1 @ X2.T) ** degree
    
    def periodic_kernel(X1, X2, period=2*np.pi):
        """Periodic kernel."""
        # For simplicity, use a single period for all dimensions
        dist_sq = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1).reshape(1, -1) - 2 * X1 @ X2.T
        dist_sq = np.maximum(dist_sq, 0)
        return np.cos(np.pi * dist_sq / period) ** 2
    
    def composite_kernel(X1, X2, rbf_ls, poly_deg, poly_coef, per_period, weights):
        """Sum of RBF, polynomial, and periodic kernels."""
        K = np.zeros((X1.shape[0], X2.shape[0]))
        if weights[0] > 0:
            K += weights[0] * rbf_kernel(X1, X2, rbf_ls)
        if weights[1] > 0:
            K += weights[1] * poly_kernel(X1, X2, poly_deg, poly_coef)
        if weights[2] > 0:
            K += weights[2] * periodic_kernel(X1, X2, per_period)
        return K
    
    def kernel_ridge_fit(X_tr, y_tr, X_val, hyperparams):
        """Fit kernel ridge regression and return predictions."""
        rbf_ls, poly_deg, poly_coef, per_period, weights, alpha = hyperparams
        K_train = composite_kernel(X_tr, X_tr, rbf_ls, poly_deg, poly_coef, per_period, weights)
        K_val = composite_kernel(X_tr, X_val, rbf_ls, poly_deg, poly_coef, per_period, weights)
        
        n_train = X_tr.shape[0]
        # Solve (K + alpha*I) w = y
        A = K_train + alpha * np.eye(n_train)
        try:
            w = np.linalg.solve(A, y_tr)
        except:
            w = np.linalg.lstsq(A, y_tr, rcond=None)[0]
        
        return K_val @ w
    
    def cross_validate_kernel(X, y, hyperparams, n_folds=5):
        """5-fold cross-validation for kernel ridge regression."""
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)
        fold_size = n // n_folds
        errors = []
        
        for fold in range(n_folds):
            val_start = fold * fold_size
            val_end = val_start + fold_size if fold < n_folds - 1 else n
            val_idx = indices[val_start:val_end]
            train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
            
            X_train_cv = X_norm[train_idx]
            y_train_cv = y[train_idx]
            X_val_cv = X_norm[val_idx]
            y_val_cv = y[val_idx]
            
            try:
                pred = kernel_ridge_fit(X_train_cv, y_train_cv, X_val_cv, hyperparams)
                mse = np.mean((pred - y_val_cv) ** 2)
                errors.append(mse)
            except:
                continue
        
        if len(errors) == 0:
            return np.inf
        return np.mean(errors)
    
    # Grid search over kernel hyperparameters
    # Length scales for RBF
    rbf_ls_options = [0.5, 1.0, 2.0, 5.0]
    # Polynomial degree
    poly_deg_options = [1, 2, 3]
    # Polynomial coef0
    poly_coef_options = [0.1, 1.0, 10.0]
    # Period for periodic kernel
    per_period_options = [2*np.pi, 4*np.pi, 10*np.pi]
    # Weights for composite kernel (normalized)
    weight_options = [
        [1.0, 0.0, 0.0],  # RBF only
        [0.0, 1.0, 0.0],  # Poly only
        [0.0, 0.0, 1.0],  # Periodic only
        [0.5, 0.5, 0.0],  # RBF + Poly
        [0.5, 0.0, 0.5],  # RBF + Periodic
        [0.0, 0.5, 0.5],  # Poly + Periodic
        [0.33, 0.33, 0.34],  # All three
    ]
    # Regularization
    alpha_options = [1e-4, 1e-3, 1e-2, 0.1, 1.0]
    
    best_cv_score = np.inf
    best_hyperparams = None
    best_predictor = None
    
    # To keep computation manageable, we'll do a coarser search
    # First, try a subset of combinations
    search_count = 0
    max_search = 200
    
    for rbf_ls in rbf_ls_options:
        for poly_deg in poly_deg_options:
            for poly_coef in poly_coef_options:
                for per_period in per_period_options:
                    for weights in weight_options:
                        for alpha in alpha_options:
                            if time.time() > deadline - 30:
                                break
                            search_count += 1
                            if search_count > max_search:
                                break
                            
                            hyperparams = (rbf_ls, poly_deg, poly_coef, per_period, weights, alpha)
                            try:
                                cv_score = cross_validate_kernel(X_norm, y, hyperparams)
                                if cv_score < best_cv_score:
                                    best_cv_score = cv_score
                                    best_hyperparams = hyperparams
                            except:
                                pass
                            if time.time() > deadline - 30:
                                break
                        if time.time() > deadline - 30:
                            break
                    if time.time() > deadline - 30:
                        break
                if time.time() > deadline - 30:
                    break
            if time.time() > deadline - 30:
                break
        if time.time() > deadline - 30:
            break
    
    # If kernel search found something, use it
    if best_hyperparams is not None:
        rbf_ls, poly_deg, poly_coef, per_period, weights, alpha = best_hyperparams
        K_train = composite_kernel(X_norm, X_norm, rbf_ls, poly_deg, poly_coef, per_period, weights)
        A = K_train + alpha * np.eye(n)
        try:
            w = np.linalg.solve(A, y)
        except:
            w = np.linalg.lstsq(A, y, rcond=None)[0]
        
        def predict(X_pred):
            X_pred_norm = (np.asarray(X_pred, dtype=np.float64) - X_mean) / X_std
            K_val = composite_kernel(X_norm, X_pred_norm, rbf_ls, poly_deg, poly_coef, per_period, weights)
            return K_val @ w
        return predict
    
    # Fallback: simple linear regression with polynomial features
    def build_poly_features(X, max_deg=3):
        cols = [np.ones(len(X))]
        for degree in range(1, max_deg + 1):
            for combo in combinations_with_replacement(range(d), degree):
                term = np.ones(len(X))
                for idx in combo:
                    term = term * X[:, idx]
                cols.append(term)
        return np.column_stack(cols)
    
    F = build_poly_features(X, 3)
    n_features = F.shape[1]
    lam = 1e-4
    A = F.T @ F + lam * np.eye(n_features)
    b = F.T @ y
    try:
        w = np.linalg.solve(A, b)
    except:
        w = np.linalg.lstsq(F, y, rcond=None)[0]
    
    def predict(X_pred):
        F_pred = build_poly_features(np.asarray(X_pred, dtype=np.float64), 3)
        return F_pred @ w
    return predict
# EVOLVE-BLOCK-END
