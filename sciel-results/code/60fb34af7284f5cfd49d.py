import numpy as np
import time

def fit(X_train, y_train):
    t0 = time.time()
    deadline = t0 + 280  # Safety margin
    
    X = np.asarray(X_train)
    y = np.asarray(y_train)
    n, d = X.shape
    
    # Since d=1 for all, we can simplify, but keep general just in case.
    # However, the problem states "1 input(s)" for all.
    
    # Normalize X to [0, 1] based on training data range
    # This helps with polynomial stability and trig frequencies
    x_min = X.min(axis=0)
    x_max = X.max(axis=0)
    x_range = x_max - x_min
    x_range[x_range < 1e-10] = 1.0
    
    def normalize(X_raw):
        return (X_raw - x_min) / x_range

    X_norm = normalize(X)
    
    # Feature Generation Function
    def generate_features(X_norm, deg_poly=6, freq_trig=3):
        # X_norm shape (n, 1)
        x = X_norm[:, 0]
        
        features = [np.ones_like(x)]
        
        # Polynomial terms
        for i in range(1, deg_poly + 1):
            features.append(x ** i)
            
        # Trigonometric terms
        # Use pi * x so that x=0 -> 0, x=1 -> pi
        # This captures one full wave over the range [0,1]
        for k in range(1, freq_trig + 1):
            features.append(np.sin(k * np.pi * x))
            features.append(np.cos(k * np.pi * x))
            
        return np.column_stack(features)

    # Define a few candidate models (feature sets)
    candidates = [
        {"deg_poly": 4, "freq_trig": 2},
        {"deg_poly": 6, "freq_trig": 3},
        {"deg_poly": 8, "freq_trig": 4},
        {"deg_poly": 10, "freq_trig": 5},
        {"deg_poly": 6, "freq_trig": 6},
    ]
    
    # Regularization strengths
    lambdas = [1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0]
    
    # Split for validation
    idx = np.arange(n)
    np.random.seed(42)
    np.random.shuffle(idx)
    split_idx = int(0.8 * n)
    train_idx = idx[:split_idx]
    val_idx = idx[split_idx:]
    
    X_train_sub = X_norm[train_idx]
    y_train_sub = y[train_idx]
    X_val_sub = X_norm[val_idx]
    y_val_sub = y[val_idx]
    
    best_val_mse = np.inf
    best_model_params = None
    best_w = None
    best_feature_config = None
    
    # Search for best model
    for config in candidates:
        if time.time() > deadline - 20:
            break
            
        F_train = generate_features(X_train_sub, config['deg_poly'], config['freq_trig'])
        F_val = generate_features(X_val_sub, config['deg_poly'], config['freq_trig'])
        
        # Ridge Regression: (F^T F + lambda I) w = F^T y
        # Add small diagonal for stability
        for lam in lambdas:
            if time.time() > deadline - 10:
                break
                
            try:
                # Efficient Ridge solution
                # w = inv(F^T F + lam I) F^T y
                # Using lstsq on augmented system is also an option, but direct solve is faster for small feature counts
                FtF = F_train.T @ F_train
                Fty = F_train.T @ y_train_sub
                A = FtF + lam * np.eye(FtF.shape[0])
                w = np.linalg.solve(A, Fty)
                
                # Evaluate on validation
                y_pred_val = F_val @ w
                mse_val = np.mean((y_pred_val - y_val_sub) ** 2)
                
                if mse_val < best_val_mse:
                    best_val_mse = mse_val
                    best_model_params = lam
                    best_feature_config = config
                    # Don't store w yet, retrain on full data later
            except np.linalg.LinAlgError:
                continue
                
    # If no model found, fallback to simple linear
    if best_model_params is None:
        best_feature_config = {"deg_poly": 2, "freq_trig": 1}
        best_model_params = 1e-6
        
    # Retrain on full training data with best config
    F_full = generate_features(X_norm, best_feature_config['deg_poly'], best_feature_config['freq_trig'])
    FtF = F_full.T @ F_full
    Fty = F_full.T @ y
    A = FtF + best_model_params * np.eye(FtF.shape[0])
    try:
        w_final = np.linalg.solve(A, Fty)
    except np.linalg.LinAlgError:
        # Fallback to lstsq
        w_final, *_ = np.linalg.lstsq(F_full, y, rcond=None)
        
    def predict(X):
        X_norm_pred = normalize(X)
        F_pred = generate_features(X_norm_pred, best_feature_config['deg_poly'], best_feature_config['freq_trig'])
        return F_pred @ w_final

    return predict
