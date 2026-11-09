"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy import linalg
from scipy.sparse import linalg as spla


def _build_feature_library(X):
    """Build a comprehensive feature library for SINDy-style sparse regression."""
    n, d = X.shape
    cols = [np.ones(n)]
    
    # Individual variable powers up to degree 5
    for j in range(d):
        xj = X[:, j]
        for p in range(1, 6):
            cols.append(xj ** p)
        # Trig features
        cols.append(np.sin(xj))
        cols.append(np.cos(xj))
        # Exponential (bounded to avoid overflow)
        cols.append(np.exp(-np.clip(xj, 0, 20)))
        cols.append(np.exp(np.clip(-xj, -20, 0)))
        # Log
        cols.append(np.log1p(np.abs(xj)))
        # Reciprocal
        cols.append(1.0 / (np.abs(xj) + 0.1))
    
    # Pairwise products
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
            cols.append(X[:, i] * X[:, j] ** 2)
            cols.append(X[:, i] ** 2 * X[:, j])
    
    # Triple products for small d
    if d <= 3:
        for i in range(d):
            for j in range(i, d):
                for k in range(j, d):
                    cols.append(X[:, i] * X[:, j] * X[:, k])
    
    return np.column_stack(cols)


def _rbf_kernel(X1, X2, length_scale):
    """Compute RBF kernel matrix between X1 and X2."""
    sq1 = np.sum(X1 ** 2, axis=1, keepdims=True)
    sq2 = np.sum(X2 ** 2, axis=1, keepdims=True).T
    sqd = sq1 + sq2 - 2.0 * (X1 @ X2.T)
    np.maximum(sqd, 0.0, out=sqd)
    return np.exp(-sqd / (2.0 * length_scale ** 2))


def _solve_ridge(K, y, alpha):
    """Solve kernel ridge regression: (K + alpha*I)^{-1} y."""
    n = K.shape[0]
    A = K + alpha * np.eye(n)
    try:
        L = np.linalg.cholesky(A)
        tmp = np.linalg.solve(L, y)
        w = np.linalg.solve(L.T, tmp)
        return w
    except np.linalg.LinAlgError:
        w, *_ = np.linalg.lstsq(A, y, rcond=None)
        return w


def _sparse_regression(X_feat, y, alpha, max_iter=100):
    """Perform sparse regression using iterative thresholding (LARS-like)."""
    n, p = X_feat.shape
    # Standardize features
    feat_mean = X_feat.mean(axis=0)
    feat_std = X_feat.std(axis=0)
    feat_std[feat_std < 1e-10] = 1.0
    X_std = (X_feat - feat_mean) / feat_std
    
    # Initialize
    w = np.zeros(p)
    y_mean = y.mean()
    y_std = y.std()
    if y_std < 1e-10:
        y_std = 1.0
    y_std_arr = (y - y_mean) / y_std
    
    # Iterative soft thresholding
    for _ in range(max_iter):
        residual = y_std_arr - X_std @ w
        # Compute correlations
        corr = X_std.T @ residual / n
        # Soft threshold
        w_new = np.sign(corr) * np.maximum(np.abs(corr) - alpha, 0.0)
        if np.allclose(w_new, w, atol=1e-6):
            break
        w = w_new
    
    # Transform back to original scale
    w_orig = w / feat_std
    w_orig[0] = (y_mean - X_feat.mean(axis=0)[1:] @ w_orig[1:]) / X_feat[:, 0].mean() if X_feat[:, 0].mean() != 0 else 0.0
    
    return w_orig


def fit(X_train, y_train):
    t0 = time.time()
    deadline = t0 + 310.0  # Leave margin for 330s total across all equations
    
    n, d = X_train.shape
    
    # Build feature library
    X_feat = _build_feature_library(X_train)
    n_feat = X_feat.shape[1]
    
    # Cross-validation setup: 5-fold
    n_folds = 5
    indices = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(indices)
    fold_indices = np.array_split(indices, n_folds)
    
    best_score = np.inf
    best_method = None
    best_params = None
    
    # Phase 1: Try sparse regression with different alpha values
    sparse_alphas = [1e-4, 1e-3, 1e-2, 0.1, 1.0, 10.0]
    
    for alpha in sparse_alphas:
        if time.time() > deadline - 30:
            break
        
        cv_scores = []
        for i in range(n_folds):
            if time.time() > deadline - 30:
                break
            val_idx = fold_indices[i]
            train_idx = np.concatenate([fold_indices[j] for j in range(n_folds) if j != i])
            
            X_train_cv = X_feat[train_idx]
            y_train_cv = y_train[train_idx]
            X_val_cv = X_feat[val_idx]
            y_val_cv = y_train[val_idx]
            
            w = _sparse_regression(X_train_cv, y_train_cv, alpha, max_iter=50)
            pred_val = X_val_cv @ w
            
            mse_val = np.mean((pred_val - y_val_cv) ** 2)
            var_y = np.var(y_val_cv)
            if var_y > 1e-12:
                nmse_val = mse_val / var_y
            else:
                nmse_val = mse_val
            cv_scores.append(nmse_val)
        
        if len(cv_scores) == n_folds:
            mean_score = np.mean(cv_scores)
            if mean_score < best_score:
                best_score = mean_score
                best_method = 'sparse'
                best_params = (alpha,)
    
    # Phase 2: Try kernel ridge regression with different hyperparameters
    length_scales = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]
    alphas_krr = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1.0]
    
    for ls in length_scales:
        if time.time() > deadline - 30:
            break
        for alpha_krr in alphas_krr:
            if time.time() > deadline - 30:
                break
            
            K_full = _rbf_kernel(X_train, X_train, ls)
            
            cv_scores = []
            for i in range(n_folds):
                if time.time() > deadline - 30:
                    break
                val_idx = fold_indices[i]
                train_idx = np.concatenate([fold_indices[j] for j in range(n_folds) if j != i])
                
                K_train = K_full[np.ix_(train_idx, train_idx)]
                y_train_cv = y_train[train_idx]
                y_val_cv = y_train[val_idx]
                
                w = _solve_ridge(K_train, y_train_cv, alpha_krr)
                K_val = K_full[np.ix_(val_idx, train_idx)]
                pred_val = K_val @ w
                
                mse_val = np.mean((pred_val - y_val_cv) ** 2)
                var_y = np.var(y_val_cv)
                if var_y > 1e-12:
                    nmse_val = mse_val / var_y
                else:
                    nmse_val = mse_val
                cv_scores.append(nmse_val)
            
            if len(cv_scores) == n_folds:
                mean_score = np.mean(cv_scores)
                if mean_score < best_score:
                    best_score = mean_score
                    best_method = 'krr'
                    best_params = (ls, alpha_krr)
    
    # Final fit with best method
    if best_method == 'sparse':
        alpha = best_params[0]
        w = _sparse_regression(X_feat, y_train, alpha, max_iter=100)
        X_feat_copy = X_feat.copy()
        
        def predict(X):
            X_test_feat = _build_feature_library(X)
            return X_test_feat @ w
        
        return predict
    
    elif best_method == 'krr':
        ls, alpha_krr = best_params
        K_full = _rbf_kernel(X_train, X_train, ls)
        w = _solve_ridge(K_full, y_train, alpha_krr)
        X_train_copy = X_train.copy()
        w_copy = w.copy()
        
        def predict(X):
            K_test = _rbf_kernel(X, X_train_copy, ls)
            return K_test @ w_copy
        
        return predict
    
    else:
        # Fallback: simple RBF with default params
        ls, alpha = 1.0, 1e-3
        K_full = _rbf_kernel(X_train, X_train, ls)
        w = _solve_ridge(K_full, y_train, alpha)
        X_train_copy = X_train.copy()
        w_copy = w.copy()
        
        def predict(X):
            K_test = _rbf_kernel(X, X_train_copy, ls)
            return K_test @ w_copy
        
        return predict
# EVOLVE-BLOCK-END
