import numpy as np
import time

_DEADLINE = time.time() + 320  # Safety margin

def _rff_kernel(X1, X2, sigma):
    """Compute RBF kernel matrix between X1 and X2."""
    X1_sq = np.sum(X1**2, axis=1)[:, np.newaxis]
    X2_sq = np.sum(X2**2, axis=1)[np.newaxis, :]
    dist_sq = X1_sq + X2_sq - 2 * X1 @ X2.T
    dist_sq = np.maximum(dist_sq, 0)
    return np.exp(-dist_sq / (2 * sigma**2))

def _kfold_split(n, k, random_state=42):
    """Simple k-fold split generator."""
    indices = np.arange(n)
    np.random.seed(random_state)
    np.random.shuffle(indices)
    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[:n % k] += 1
    current = 0
    for fold_size in fold_sizes:
        yield indices[current:current + fold_size]
        current += fold_size

def fit(X_train, y_train):
    d = X_train.shape[1]
    n_samples = X_train.shape[0]
    
    # Determine input scale
    ranges = np.max(X_train, axis=0) - np.min(X_train, axis=0)
    valid_ranges = ranges[ranges > 0]
    avg_range = np.mean(valid_ranges) if len(valid_ranges) > 0 else 1.0
    
    # Candidate hyperparameters
    # Sigma: length scale. Related to input range.
    sigma_candidates = np.logspace(-1, 2, 8) * avg_range
    # Alpha: regularization
    alpha_candidates = np.logspace(-3, 3, 8)
    
    best_cv_mse = np.inf
    best_sigma = sigma_candidates[len(sigma_candidates)//2]
    best_alpha = alpha_candidates[len(alpha_candidates)//2]
    
    n_folds = 3
    folds = list(_kfold_split(n_samples, n_folds))
    
    for sigma in sigma_candidates:
        if time.time() > _DEADLINE - 10:
            break
        for alpha in alpha_candidates:
            if time.time() > _DEADLINE - 10:
                break
            
            fold_mses = []
            for i in range(n_folds):
                val_idx = folds[i]
                train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
                
                X_tr = X_train[train_idx]
                y_tr = y_train[train_idx]
                X_val = X_train[val_idx]
                y_val = y_train[val_idx]
                
                K_train = _rff_kernel(X_tr, X_tr, sigma)
                K_val = _rff_kernel(X_val, X_tr, sigma)
                
                I = np.eye(len(X_tr))
                try:
                    w = np.linalg.solve(K_train + alpha * I, y_tr)
                except np.linalg.LinAlgError:
                    w = np.linalg.lstsq(K_train + alpha * I, y_tr, rcond=None)[0]
                
                pred_val = K_val @ w
                mse = np.mean((pred_val - y_val)**2)
                fold_mses.append(mse)
            
            mean_cv_mse = np.mean(fold_mses)
            if mean_cv_mse < best_cv_mse:
                best_cv_mse = mean_cv_mse
                best_sigma = sigma
                best_alpha = alpha
    
    # Refit on full training data
    K_train = _rff_kernel(X_train, X_train, best_sigma)
    I = np.eye(n_samples)
    try:
        w_final = np.linalg.solve(K_train + best_alpha * I, y_train)
    except np.linalg.LinAlgError:
        w_final = np.linalg.lstsq(K_train + best_alpha * I, y_train, rcond=None)[0]
    
    X_train_ref = X_train
    w_final_ref = w_final
    sigma_ref = best_sigma
    
    def predict(X):
        K_test = _rff_kernel(X, X_train_ref, sigma_ref)
        return K_test @ w_final_ref
    
    return predict
