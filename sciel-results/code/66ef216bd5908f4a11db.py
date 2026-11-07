"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import cdist

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 320.0  # 10s safety margin before 330s kill
    
    X = X_train.astype(np.float64)
    y = y_train.astype(np.float64)
    n = X.shape[0]
    d = X.shape[1]
    
    # Normalize X for better numerical conditioning
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std
    
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std == 0:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    def matern52_kernel(X1, X2, length_scale, signal_var):
        """Matérn 5/2 kernel."""
        d1 = cdist(X1, X2)
        z = np.sqrt(3.0) * d1 / length_scale
        K = signal_var * (1 + z + z**2 / 3.0) * np.exp(-z)
        return K
    
    # 5-fold cross-validation for hyperparameter selection
    def cross_validate(length_scales, signal_vars, n_folds=5):
        if time.time() > deadline - 5:
            return float('inf')
        
        indices = np.arange(n)
        np.random.seed(42)
        np.random.shuffle(indices)
        fold_size = n // n_folds
        folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(n_folds)]
        
        best_score = float('inf')
        
        for ls in length_scales:
            for sv in signal_vars:
                if time.time() > deadline - 5:
                    return best_score
                cv_errors = []
                for i in range(n_folds):
                    val_idx = folds[i]
                    train_idx = np.setdiff1d(indices, val_idx)
                    
                    X_tr = X_norm[train_idx]
                    y_tr = y_norm[train_idx]
                    X_val = X_norm[val_idx]
                    y_val = y_norm[val_idx]
                    
                    K_train = matern52_kernel(X_tr, X_tr, ls, sv)
                    K_train += 1e-10 * np.eye(len(X_tr))
                    K_val_train = matern52_kernel(X_val, X_tr, ls, sv)
                    
                    try:
                        L = np.linalg.cholesky(K_train)
                        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_tr))
                        pred = K_val_train @ alpha
                        cv_errors.append(np.mean((pred - y_val) ** 2))
                    except np.linalg.LinAlgError:
                        cv_errors.append(float('inf'))
                
                score = np.mean(cv_errors)
                if score < best_score:
                    best_score = score
        
        return best_score
    
    # Grid search over hyperparameters
    length_scales = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    signal_vars = [0.1, 0.5, 1.0, 2.0, 5.0]
    
    best_score = float('inf')
    best_ls = 1.0
    best_sv = 1.0
    
    for ls in length_scales:
        for sv in signal_vars:
            if time.time() > deadline - 30:
                break
            score = cross_validate([ls], [sv])
            if score < best_score:
                best_score = score
                best_ls = ls
                best_sv = sv
    
    # Fit final GP with best hyperparameters
    K_train = matern52_kernel(X_norm, X_norm, best_ls, best_sv)
    K_train += 1e-10 * np.eye(n)
    
    try:
        L = np.linalg.cholesky(K_train)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
    except np.linalg.LinAlgError:
        # Fallback: use ridge regression if GP fails
        def features(X):
            cols = [np.ones(len(X))] + [X[:, j] for j in range(d)]
            for deg in range(2, 5):
                for i in range(d):
                    cols.append(X[:, i] ** deg)
                for i in range(d):
                    for j in range(i + 1, d):
                        cols.append(X[:, i] ** (deg - 1) * X[:, j])
            return np.column_stack(cols)
        w, *_ = np.linalg.lstsq(features(X_train), y_train, rcond=None)
        def predict(X):
            X = np.asarray(X, dtype=np.float64)
            return features(X) @ w
        return predict
    
    def predict(X):
        X = np.asarray(X, dtype=np.float64)
        X_pred_norm = (X - X_mean) / X_std
        K_pred_train = matern52_kernel(X_pred_norm, X_norm, best_ls, best_sv)
        y_pred_norm = K_pred_train @ alpha
        return y_pred_norm * y_std + y_mean
    
    return predict
# EVOLVE-BLOCK-END
