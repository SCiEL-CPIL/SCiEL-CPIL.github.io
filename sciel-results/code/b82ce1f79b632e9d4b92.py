import numpy as np

def fit(X_train, y_train):
    X_train = np.asarray(X_train).flatten()
    y_train = np.asarray(y_train).flatten()
    
    # Sort and handle duplicates (average y for same x)
    sort_idx = np.argsort(X_train)
    X_sorted = X_train[sort_idx]
    y_sorted = y_train[sort_idx]
    
    unique_x, inverse_indices, counts = np.unique(X_sorted, return_inverse=True, return_counts=True)
    if len(unique_x) < len(X_sorted):
        y_sum = np.zeros_like(unique_x)
        np.add.at(y_sum, inverse_indices, y_sorted)
        y_avg = y_sum / counts
        X_fit = unique_x
        y_fit = y_avg
    else:
        X_fit = X_sorted
        y_fit = y_sorted
        
    n = len(X_fit)
    
    # Estimate kernel width sigma
    # Use median distance between points as a heuristic
    if n > 1:
        diffs = np.diff(X_fit)
        median_diff = np.median(diffs)
        if median_diff == 0:
            median_diff = 1.0
        sigma = 1.0 / (median_diff * 2.0) # Heuristic
    else:
        sigma = 1.0
        
    # Build RBF Kernel Matrix
    # K_ij = exp(-sigma^2 * (x_i - x_j)^2)
    diff = X_fit[:, np.newaxis] - X_fit[np.newaxis, :]
    K = np.exp(-sigma**2 * diff**2)
    
    # Add small regularization to handle potential ill-conditioning, 
    # even though noiseless, numerical stability is key.
    lam = 1e-8
    K_reg = K + lam * np.eye(n)
    
    try:
        alpha = np.linalg.solve(K_reg, y_fit)
    except np.linalg.LinAlgError:
        # Fallback to lstsq if singular
        alpha = np.linalg.lstsq(K_reg, y_fit, rcond=None)[0]
        
    def predict(X):
        X = np.asarray(X).flatten()
        # Compute kernel matrix between test points and training points
        diff_pred = X[:, np.newaxis] - X_fit[np.newaxis, :]
        K_pred = np.exp(-sigma**2 * diff_pred**2)
        return K_pred @ alpha

    return predict
