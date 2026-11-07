import numpy as np
from scipy.spatial.distance import cdist

def fit(X_train, y_train):
    X = np.asarray(X_train, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.asarray(y_train, dtype=float)
    
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    y = np.nan_to_num(y, nan=0.0, posinf=1e6, neginf=-1e6)
    
    n_samples, n_features = X.shape
    
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_std[X_std == 0] = 1.0
    X_norm = (X - X_mean) / X_std
    
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std == 0:
        y_std = 1.0
    y_norm = (y - y_mean) / y_std
    
    # Precompute distance matrix for training set
    D_train = cdist(X_norm, X_norm, metric='euclidean')
    
    # Grid of K values
    K_grid = [1, 3, 5, 7, 11, 15, 21, 31, 51, 101, 151]
    
    # Cross-validation setup
    n_folds = 5
    indices = np.arange(n_samples)
    np.random.seed(42)
    np.random.shuffle(indices)
    folds = np.array_split(indices, n_folds)
    
    best_mse = np.inf
    best_K = 5
    
    for K in K_grid:
        if K >= n_samples:
            continue
            
        cv_mse_sum = 0.0
        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])
            
            # Distances from validation points to training points
            D_val_train = D_train[np.ix_(val_idx, train_idx)]
            y_train_fold = y_norm[train_idx]
            y_val_fold = y_norm[val_idx]
            
            # For each validation point, find K nearest neighbors
            sorted_indices = np.argsort(D_val_train, axis=1)
            nearest_indices = sorted_indices[:, :K]
            
            # Get distances to nearest neighbors
            row_indices = np.arange(len(val_idx))[:, np.newaxis]
            nearest_dists = D_val_train[row_indices, nearest_indices]
            
            # Inverse distance weighting
            eps = 1e-8
            weights = 1.0 / (nearest_dists + eps)
            weight_sums = np.sum(weights, axis=1, keepdims=True)
            weights = weights / weight_sums
            
            # Weighted average
            y_nearest = y_train_fold[nearest_indices]
            y_pred_val = np.sum(weights * y_nearest, axis=1)
            
            mse = np.mean((y_val_fold - y_pred_val)**2)
            cv_mse_sum += mse
            
        mean_cv_mse = cv_mse_sum / n_folds
        if mean_cv_mse < best_mse:
            best_mse = mean_cv_mse
            best_K = K
            
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=float)
        if X_test.ndim == 1:
            X_test = X_test.reshape(1, -1)
        
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=1e6, neginf=-1e6)
        X_test_norm = (X_test - X_mean) / X_std
        
        # Distances from test points to all training points
        D_test_train = cdist(X_test_norm, X_norm, metric='euclidean')
        
        # Find best_K nearest neighbors
        sorted_indices = np.argsort(D_test_train, axis=1)
        nearest_indices = sorted_indices[:, :best_K]
        
        # Get distances
        row_indices = np.arange(len(X_test))[:, np.newaxis]
        nearest_dists = D_test_train[row_indices, nearest_indices]
        
        # Inverse distance weighting
        eps = 1e-8
        weights = 1.0 / (nearest_dists + eps)
        weight_sums = np.sum(weights, axis=1, keepdims=True)
        weights = weights / weight_sums
        
        # Weighted average
        y_nearest = y_norm[nearest_indices]
        y_pred_norm = np.sum(weights * y_nearest, axis=1)
        
        y_pred = y_pred_norm * y_std + y_mean
        y_pred = np.nan_to_num(y_pred, nan=0.0, posinf=1e6, neginf=-1e6)
        
        return y_pred

    return predict
