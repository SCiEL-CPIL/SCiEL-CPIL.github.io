"""Rewritten program for symbolic regression task."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools

def _generate_features(X):
    """
    Generate a library of candidate features for symbolic regression.
    X: numpy array of shape (n_samples, n_features)
    Returns: numpy array of shape (n_samples, n_candidates)
    """
    n, d = X.shape
    features = [np.ones(n)]  # Intercept
    
    # Helper to add a feature
    def add_feat(f):
        # Ensure no NaNs or infs
        if np.all(np.isfinite(f)):
            features.append(f)
    
    # 1. Polynomial terms up to degree 4
    # For each dimension
    for i in range(d):
        xi = X[:, i]
        add_feat(xi)
        add_feat(xi**2)
        add_feat(xi**3)
        add_feat(xi**4)
        
    # Interactions
    for i in range(d):
        for j in range(i, d):
            if i == j:
                continue # Already handled in power terms
            xi = X[:, i]
            xj = X[:, j]
            add_feat(xi * xj)
            add_feat(xi * xj**2)
            add_feat(xi**2 * xj)
            add_feat(xi**2 * xj**2)
            
    # Triple interactions if d >= 3
    if d >= 3:
        for i in range(d):
            for j in range(i+1, d):
                for k in range(j+1, d):
                    xi = X[:, i]
                    xj = X[:, j]
                    xk = X[:, k]
                    add_feat(xi * xj * xk)
                    add_feat(xi**2 * xj * xk)
                    add_feat(xi * xj**2 * xk)
                    add_feat(xi * xj * xk**2)

    # 2. Trigonometric and Exponential terms
    for i in range(d):
        xi = X[:, i]
        # Clip to avoid extreme values in exp if range is large, but range is [1,10]
        add_feat(np.sin(xi))
        add_feat(np.cos(xi))
        add_feat(np.exp(-xi))
        add_feat(np.exp(xi / 10.0)) # Scaled to avoid overflow
        add_feat(np.log(xi + 1.0))
        
    # Interactions of trig/exp
    for i in range(d):
        for j in range(i+1, d):
            xi = X[:, i]
            xj = X[:, j]
            add_feat(np.sin(xi) * np.cos(xj))
            add_feat(np.exp(-xi) * np.exp(-xj))
            add_feat(np.sin(xi + xj))
            add_feat(np.cos(xi - xj))
            
    # Stack
    F = np.column_stack(features)
    # Remove duplicate columns (e.g., if sin(x) is constant or linearly dependent)
    # Simple check: rank
    # We don't strictly need to remove duplicates for lstsq, but it speeds up selection slightly.
    # However, keeping them is safer for diversity.
    return F

def fit(X_train, y_train):
    start_time = time.time()
    timeout = 25.0 # 25s per equation, leaving 5s buffer for 10 eqs
    
    X = np.asarray(X_train, dtype=np.float64)
    y = np.asarray(y_train, dtype=np.float64)
    
    # Generate feature library
    F = _generate_features(X)
    n_samples, n_features = F.shape
    
    # Forward Selection
    # Start with intercept only (index 0)
    selected = [0]
    current_mse = np.var(y) # MSE of predicting mean
    
    max_steps = 20 # Limit number of features to prevent overfitting and save time
    
    for _ in range(max_steps):
        if time.time() - start_time > timeout:
            break
            
        best_idx = -1
        best_mse = current_mse
        
        for j in range(1, n_features):
            if j in selected:
                continue
                
            # Try adding feature j
            new_selected = selected + [j]
            F_sub = F[:, new_selected]
            
            # Solve OLS
            try:
                w, *_ = np.linalg.lstsq(F_sub, y, rcond=None)
                y_pred = F_sub @ w
                mse = np.mean((y - y_pred)**2)
                
                if mse < best_mse - 1e-6: # Significant improvement
                    best_mse = mse
                    best_idx = j
            except np.linalg.LinAlgError:
                continue
        
        if best_idx == -1:
            break
            
        selected.append(best_idx)
        current_mse = best_mse
        
        # Optional: Early stop if MSE is very low
        if current_mse < 1e-8:
            break
            
    # Final fit with selected features
    F_final = F[:, selected]
    w_final, *_ = np.linalg.lstsq(F_final, y, rcond=None)
    
    # Create predict function
    def predict(X_test):
        X_test = np.asarray(X_test, dtype=np.float64)
        F_test = _generate_features(X_test)
        F_test_sub = F_test[:, selected]
        return F_test_sub @ w_final
        
    return predict
# EVOLVE-BLOCK-END
