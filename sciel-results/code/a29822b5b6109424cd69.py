"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    # Set a hard deadline to ensure we return something
    # 300s total for 4 equations. Give 280s for safety.
    # This fit function is called 4 times.
    # We don't know how much time has passed in previous calls.
    # But usually, these are fast.
    # Let's just make it efficient.
    
    x = X_train[:, 0].flatten()
    y = y_train.flatten()
    n = len(x)
    
    if n == 0:
        return lambda X: np.zeros((len(X), 1))
        
    # If all y are same, return constant
    if np.allclose(y, y[0]):
        return lambda X: np.ones((len(X), 1)) * y[0]

    # 1. Split for Cross-Validation
    indices = np.random.permutation(n)
    val_size = 20
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]
    
    x_train_cv = x[train_idx]
    y_train_cv = y[train_idx]
    x_val_cv = x[val_idx]
    y_val_cv = y[val_idx]
    
    # 2. Define candidate sigmas
    x_min, x_max = np.min(x), np.max(x)
    x_range = x_max - x_min
    if x_range < 1e-6:
        x_range = 1.0
        
    # Log-spaced sigmas
    # Lower bound: small enough to capture local features
    # Upper bound: large enough to be global
    sigmas = np.logspace(np.log10(x_range * 0.005), np.log10(x_range * 5.0), 60)
    
    best_sigma = sigmas[0]
    best_val_mse = np.inf
    best_w = None
    
    # 3. Search for best sigma
    for sigma in sigmas:
        # Compute kernel for training set (CV train)
        diff_train = x_train_cv[:, np.newaxis] - x_train_cv[np.newaxis, :]
        K_train = np.exp(-(diff_train ** 2) / (2 * sigma ** 2))
        
        # Regularization
        # Use a small ridge to stabilize
        lam = 1e-5
        try:
            # Solve (K + lam I) w = y
            # Using lstsq is safer than inv
            w, residuals, rank, s = np.linalg.lstsq(K_train + lam * np.eye(len(x_train_cv)), y_train_cv, rcond=None)
            
            # Predict on validation set
            diff_val = x_val_cv[:, np.newaxis] - x_train_cv[np.newaxis, :]
            K_val = np.exp(-(diff_val ** 2) / (2 * sigma ** 2))
            y_pred_val = K_val @ w
            
            val_mse = np.mean((y_pred_val - y_val_cv) ** 2)
            
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_sigma = sigma
                best_w = w # Store weights? No, we need to re-fit on all data.
                
        except Exception:
            continue

    # 4. Re-fit on all data with best sigma
    diff_all = x[:, np.newaxis] - x[np.newaxis, :]
    K_all = np.exp(-(diff_all ** 2) / (2 * best_sigma ** 2))
    
    lam = 1e-5
    try:
        w_final, residuals, rank, s = np.linalg.lstsq(K_all + lam * np.eye(n), y, rcond=None)
    except Exception:
        # Fallback to mean
        mean_y = np.mean(y)
        return lambda X: np.ones((len(X), 1)) * mean_y

    # 5. Define predictor
    def predict(X):
        x_test = X[:, 0].flatten()
        diff_test = x_test[:, np.newaxis] - x[np.newaxis, :]
        K_test = np.exp(-(diff_test ** 2) / (2 * best_sigma ** 2))
        y_pred = K_test @ w_final
        return y_pred.reshape(-1, 1)
        
    return predict
# EVOLVE-BLOCK-END
