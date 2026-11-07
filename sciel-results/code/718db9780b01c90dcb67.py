import numpy as np

def fit(X_train, y_train):
    X_train = np.asarray(X_train).flatten()
    y_train = np.asarray(y_train).flatten()
    
    # Handle edge case: constant y or constant X
    if np.allclose(y_train, y_train[0]):
        c = y_train[0]
        def predict(X):
            return np.full(np.asarray(X).shape, c)
        return predict
        
    if np.allclose(X_train, X_train[0]):
        c = np.mean(y_train)
        def predict(X):
            return np.full(np.asarray(X).shape, c)
        return predict

    # Scale X to [-1, 1] for better numerical stability
    x_min, x_max = np.min(X_train), np.max(X_train)
    if x_max == x_min:
        x_scale = 1.0
        x_offset = 0.0
    else:
        x_scale = 2.0 / (x_max - x_min)
        x_offset = x_min
        
    X_scaled = (X_train - x_offset) * x_scale
    
    # Split for model selection
    idx = np.random.permutation(len(X_train))
    split_idx = int(0.8 * len(X_train))
    X_val, y_val = X_scaled[idx[split_idx:]], y_train[idx[split_idx:]]
    X_tr, y_tr = X_scaled[idx[:split_idx]], y_train[idx[:split_idx]]
    
    best_degree = 2
    best_mse = np.inf
    
    degrees = [2, 4, 6, 8, 10, 12, 15, 18]
    
    for D in degrees:
        # Construct Vandermonde matrix
        V_tr = np.vander(X_tr, N=D+1, increasing=True)
        V_val = np.vander(X_val, N=D+1, increasing=True)
        
        # Solve least squares
        try:
            theta, _, _, _ = np.linalg.lstsq(V_tr, y_tr, rcond=None)
            y_pred_val = V_val @ theta
            mse = np.mean((y_pred_val - y_val) ** 2)
            
            if mse < best_mse:
                best_mse = mse
                best_degree = D
        except Exception:
            continue
            
    # Refit on all data with best degree
    V_full = np.vander(X_scaled, N=best_degree+1, increasing=True)
    theta_full, _, _, _ = np.linalg.lstsq(V_full, y_train, rcond=None)
    
    def predict(X):
        X = np.asarray(X).flatten()
        X_scaled_test = (X - x_offset) * x_scale
        V_test = np.vander(X_scaled_test, N=best_degree+1, increasing=True)
        return V_test @ theta_full

    return predict
