"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.interpolate import PchipInterpolator

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290  # Safety margin for 330s limit
    
    # Flatten X for 1D processing
    X = np.asarray(X_train, dtype=float).reshape(-1, 1)
    y = np.asarray(y_train, dtype=float).flatten()
    n = len(y)
    
    # Normalize X to [-1, 1] for better polynomial conditioning
    x_min = np.min(X)
    x_max = np.max(X)
    x_range = x_max - x_min
    if x_range < 1e-10:
        x_range = 1.0
    
    def normalize(x):
        return 2 * (x - x_min) / x_range - 1
        
    X_norm = normalize(X)
    
    # Try several polynomial degrees and pick the best one based on training error
    best_w = None
    best_degree = 10
    best_train_error = np.inf
    
    degrees_to_try = [10, 15, 20, 25, 30, 40]
    
    for deg in degrees_to_try:
        if time.time() > deadline:
            break
            
        # Create Vandermonde matrix
        V = np.vander(X_norm.flatten(), N=deg+1, increasing=True)
        
        # Solve least squares
        try:
            w, residuals, rank, s = np.linalg.lstsq(V, y, rcond=None)
            
            # Calculate training error
            y_pred = V @ w
            train_error = np.mean((y_pred - y)**2)
            
            if train_error < best_train_error:
                best_train_error = train_error
                best_w = w
                best_degree = deg
        except Exception:
            continue
            
    # Try Spline Interpolation
    spline = None
    try:
        sort_idx = np.argsort(X.flatten())
        X_sorted = X.flatten()[sort_idx]
        y_sorted = y[sort_idx]
        
        spline = PchipInterpolator(X_sorted, y_sorted)
        y_pred_spline = spline(X.flatten())
        spline_error = np.mean((y_pred_spline - y)**2)
        
        # If spline error is significantly better or polynomial is bad, use spline
        if spline_error < best_train_error * 0.5 or best_train_error > 1e-6:
             def predict(X_pred):
                 X_pred = np.asarray(X_pred, dtype=float).flatten()
                 return spline(X_pred)
             return predict
    except Exception:
        pass
        
    # Fallback to polynomial
    def predict(X_pred):
        X_pred = np.asarray(X_pred, dtype=float).flatten()
        X_pred_norm = normalize(X_pred)
        V_pred = np.vander(X_pred_norm, N=best_degree+1, increasing=True)
        return V_pred @ best_w

    return predict
# EVOLVE-BLOCK-END
