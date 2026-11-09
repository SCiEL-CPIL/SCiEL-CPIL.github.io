import numpy as np
import time
from scipy.spatial.distance import cdist

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 60.0  # 60s per equation, 4*60=240s < 300s limit
    
    X = X_train
    y = y_train
    n = X.shape[0]
    
    # Standardize X
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    # Avoid division by zero
    X_std[X_std == 0] = 1.0
    X_scaled = (X - X_mean) / X_std
    
    y_mean = np.mean(y)
    y_std = np.std(y)
    if y_std == 0:
        y_std = 1.0
    y_scaled = (y - y_mean) / y_std
    
    # Split for validation
    np.random.seed(42)
    indices = np.random.permutation(n)
    val_size = max(10, n // 5)
    val_idx = indices[:val_size]
    tr_idx = indices[val_size:]
    
    X_tr = X_scaled[tr_idx]
    y_tr = y_scaled[tr_idx]
    X_val = X_scaled[val_idx]
    y_val = y_scaled[val_idx]
    
    # Hyperparameter grid for RBF Kernel
    # K(x, x') = exp(-||x - x'||^2 / (2 * l^2))
    length_scales = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    alphas = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    
    best_val_nmse = np.inf
    best_l = 1.0
    best_alpha = 1e-3
    best_w = None
    best_K_inv = None
    
    # Pre-compute distances for training set to speed up kernel construction
    # K_tr = exp(-D_tr^2 / (2*l^2))
    # We can compute D_tr^2 once
    D_tr_sq = cdist(X_tr, X_tr, metric='sqeuclidean')
    
    # We also need distances for validation set to training set
    D_val_tr_sq = cdist(X_val, X_tr, metric='sqeuclidean')
    
    for l in length_scales:
        if time.time() > deadline - 5:
            break
        inv_2l2 = 1.0 / (2.0 * l * l)
        
        # K_tr for this length scale
        K_tr = np.exp(-D_tr_sq * inv_2l2)
        # K_val_tr for this length scale
        K_val_tr = np.exp(-D_val_tr_sq * inv_2l2)
        
        for alpha in alphas:
            if time.time() > deadline - 5:
                break
            
            # Solve (K + alpha*I) w = y
            # Using Cholesky or direct solve
            try:
                K_reg = K_tr + alpha * np.eye(len(y_tr))
                # w = K_reg^-1 y_tr
                w = np.linalg.solve(K_reg, y_tr)
                
                # Predict on validation set
                y_val_pred = K_val_tr @ w
                
                # Calculate NMSE on validation set
                # NMSE = mean((pred - y)^2) / var(y)
                # Note: y_val is scaled, so var(y_val) = 1. 
                # NMSE_scaled = mean((pred - y_val)^2)
                # This is equivalent to NMSE_original because scaling cancels out in ratio.
                val_mse = np.mean((y_val_pred - y_val) ** 2)
                
                if val_mse < best_val_nmse:
                    best_val_nmse = val_mse
                    best_l = l
                    best_alpha = alpha
                    best_w = w
                    # Store K_tr inverse or just refit later? 
                    # Storing w is not enough if we refit on full data.
                    # We just store hyperparameters and refit.
            except np.linalg.LinAlgError:
                continue

    # Refit best model on all training data
    if time.time() > deadline - 2:
        # Fallback: simple linear regression
        def predict(X):
            X_new = (X - X_mean) / X_std
            # Linear fit on scaled X
            # Use polyfit or lstsq
            # Since d=1, use polyfit degree 1
            coeffs = np.polyfit(X_new[:, 0], y_scaled, 1)
            pred_scaled = np.polyval(coeffs, X_new[:, 0])
            return pred_scaled * y_std + y_mean
        return predict

    inv_2l2_best = 1.0 / (2.0 * best_l * best_l)
    D_full_sq = cdist(X_scaled, X_scaled, metric='sqeuclidean')
    K_full = np.exp(-D_full_sq * inv_2l2_best)
    
    try:
        K_full_reg = K_full + best_alpha * np.eye(n)
        w_full = np.linalg.solve(K_full_reg, y_scaled)
    except np.linalg.LinAlgError:
        # Fallback to linear
        def predict(X):
            X_new = (X - X_mean) / X_std
            coeffs = np.polyfit(X_new[:, 0], y_scaled, 1)
            pred_scaled = np.polyval(coeffs, X_new[:, 0])
            return pred_scaled * y_std + y_mean
        return predict

    # Capture variables in closure
    K_full_reg_inv = np.linalg.inv(K_full_reg)
    
    def predict(X):
        X_new = (X - X_mean) / X_std
        # Compute kernel between X_new and X_scaled
        D_new_sq = cdist(X_new, X_scaled, metric='sqeuclidean')
        K_new = np.exp(-D_new_sq * inv_2l2_best)
        
        # Prediction: K_new @ K_full_reg^-1 @ y_scaled
        # Pre-computing K_full_reg^-1 is O(n^3) but done once.
        # Alternatively, solve (K_full + alpha I) v = K_new.T each time? No, that's O(n^3) per batch.
        # Better: Pre-compute alpha_inv = (K_full + alpha I)^-1.
        # Then pred = K_new @ (alpha_inv @ y_scaled).
        # Let's pre-compute the solution vector for the training targets.
        pass 

    # Actually, let's just store the solution vector for the training set
    # y_pred = K(X, X_train) @ w_full
    # w_full is already computed.
    
    def predict(X):
        X_new = (X - X_mean) / X_std
        D_new_sq = cdist(X_new, X_scaled, metric='sqeuclidean')
        K_new = np.exp(-D_new_sq * inv_2l2_best)
        pred_scaled = K_new @ w_full
        return pred_scaled * y_std + y_mean

    return predict
