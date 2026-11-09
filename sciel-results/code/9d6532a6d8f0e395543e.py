import numpy as np
import time

def fit(X_train, y_train):
    t_start = time.time()
    deadline = t_start + 320.0
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    n, d = X.shape
    
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-10] = 1.0
    X_std = (X - mu) / sigma
    
    def standardize(X_new):
        return (np.asarray(X_new, dtype=np.float64) - mu) / sigma
        
    def matern_kernel(X1, X2, l):
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
        np.maximum(sq_dists, 0, out=sq_dists)
        r = np.sqrt(sq_dists) / l
        return (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)

    # Hyperparameter search: Length scale
    # We use a small alpha for near-interpolation
    alpha = 1e-6
    
    best_l = 1.0
    best_cv_error = np.inf
    
    # Simple K-Fold CV (5 folds)
    n_folds = 5
    indices = np.arange(n)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    candidate_ls = [0.5, 1.0, 2.0, 4.0]
    
    for l in candidate_ls:
        fold_errors = []
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            
            X_tr = X_std[train_idx]
            y_tr = y[train_idx]
            X_val = X_std[val_idx]
            y_val = y[val_idx]
            
            K_tr = matern_kernel(X_tr, X_tr, l)
            K_reg = K_tr + alpha * np.eye(len(X_tr))
            try:
                w = np.linalg.solve(K_reg, y_tr)
            except:
                w, *_ = np.linalg.lstsq(K_reg, y_tr, rcond=None)
                
            K_val = matern_kernel(X_val, X_tr, l)
            pred_val = K_val @ w
            err = np.mean((pred_val - y_val)**2)
            fold_errors.append(err)
            
        mean_err = np.mean(fold_errors)
        if mean_err < best_cv_error:
            best_cv_error = mean_err
            best_l = l
            
    # Final fit with best_l
    K = matern_kernel(X_std, X_std, best_l)
    K_reg = K + alpha * np.eye(n)
    try:
        w = np.linalg.solve(K_reg, y)
    except:
        w, *_ = np.linalg.lstsq(K_reg, y, rcond=None)

    def predict(X_test):
        X_test_std = standardize(X_test)
        K_test = matern_kernel(X_test_std, X_std, best_l)
        return K_test @ w

    return predict
