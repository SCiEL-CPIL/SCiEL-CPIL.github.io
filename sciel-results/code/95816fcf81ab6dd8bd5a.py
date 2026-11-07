import numpy as np
import time

def fit(X_train, y_train):
    start_time = time.time()
    deadline = start_time + 290
    
    X_flat = X_train.flatten()
    y_flat = y_train.flatten()
    
    sort_idx = np.argsort(X_flat)
    X_sorted = X_flat[sort_idx]
    y_sorted = y_flat[sort_idx]
    
    unique_X, inverse_indices, counts = np.unique(X_sorted, return_inverse=True, return_counts=True)
    if len(unique_X) != len(X_sorted):
        y_sum = np.zeros_like(unique_X, dtype=float)
        np.add.at(y_sum, inverse_indices, y_sorted)
        y_sorted = y_sum / counts
        X_sorted = unique_X
        
    if len(X_sorted) < 4:
        mean_y = np.mean(y_flat)
        return lambda X: np.full_like(X.flatten(), mean_y).reshape(X.shape)

    x_min, x_max = X_sorted[0], X_sorted[-1]
    if x_max == x_min:
        return lambda X: np.full_like(X.flatten(), y_sorted[0]).reshape(X.shape)
    
    # Normalize to [-1, 1] for Chebyshev stability
    X_norm = (X_sorted - (x_max + x_min)/2) / ((x_max - x_min)/2)
    
    best_c = None
    best_deg = 10
    min_err = float('inf')
    
    n = len(X_sorted)
    split = int(n * 0.9)
    X_tr, y_tr = X_norm[:split], y_sorted[:split]
    X_val, y_val = X_norm[split:], y_sorted[split:]
    
    for deg in [10, 20, 30, 50, 80, 100]:
        if time.time() > deadline:
            break
        try:
            c = np.polynomial.chebyshev.chebfit(X_tr, y_tr, deg)
            y_pred = np.polynomial.chebyshev.chebval(X_val, c)
            err = np.mean((y_pred - y_val)**2)
            if err < min_err:
                min_err = err
                best_c = c
                best_deg = deg
        except Exception:
            continue
            
    if best_c is None:
        try:
            best_c = np.polynomial.chebyshev.chebfit(X_norm, y_sorted, 10)
            best_deg = 10
        except:
            mean_y = np.mean(y_sorted)
            return lambda X: np.full_like(X.flatten(), mean_y).reshape(X.shape)
            
    try:
        c_final = np.polynomial.chebyshev.chebfit(X_norm, y_sorted, best_deg)
    except Exception:
        c_final = best_c

    def predict(X):
        X_eval = np.asarray(X).flatten()
        X_eval_norm = (X_eval - (x_max + x_min)/2) / ((x_max - x_min)/2)
        y_pred = np.polynomial.chebyshev.chebval(X_eval_norm, c_final)
        return y_pred.reshape(np.asarray(X).shape)

    return predict
