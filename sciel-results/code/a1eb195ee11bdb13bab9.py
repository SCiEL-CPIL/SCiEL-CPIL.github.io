# EVOLVE-BLOCK-START
import numpy as np
import time

def _generate_symbolic_features(X):
    """
    Generate a rich set of candidate symbolic features for 1D input.
    X is shape (n, 1).
    """
    n = X.shape[0]
    x = X[:, 0]
    
    # Clip x to avoid overflow in exp
    x_clip = np.clip(x, -10, 10)
    
    features = []
    names = []
    
    # Constant
    features.append(np.ones(n))
    names.append('1')
    
    # Polynomials
    for p in range(1, 8):
        features.append(x**p)
        names.append(f'x^{p}')
        
    # Trig
    features.append(np.sin(x))
    names.append('sin(x)')
    features.append(np.cos(x))
    names.append('cos(x)')
    features.append(np.sin(2*x))
    names.append('sin(2x)')
    features.append(np.cos(2*x))
    names.append('cos(2x)')
    features.append(np.sin(x**2))
    names.append('sin(x^2)')
    features.append(np.cos(x**2))
    names.append('cos(x^2)')
    
    # Exp
    features.append(np.exp(x_clip))
    names.append('exp(x)')
    features.append(np.exp(-x_clip))
    names.append('exp(-x)')
    features.append(np.exp(x_clip**2)) # Might be large, but useful
    names.append('exp(x^2)')
    features.append(np.exp(-x_clip**2))
    names.append('exp(-x^2)')
    
    # Log
    features.append(np.log1p(np.abs(x)))
    names.append('log1p(|x|)')
    features.append(np.sign(x) * np.log1p(np.abs(x)))
    names.append('sign(x)*log1p(|x|)')
    
    # Hyperbolic
    features.append(np.tanh(x_clip))
    names.append('tanh(x)')
    features.append(np.sinh(x_clip))
    names.append('sinh(x)')
    features.append(np.cosh(x_clip))
    names.append('cosh(x)')
    
    # Combinations
    features.append(x * np.sin(x))
    names.append('x*sin(x)')
    features.append(x * np.cos(x))
    names.append('x*cos(x)')
    features.append(x * np.exp(-x_clip))
    names.append('x*exp(-x)')
    features.append(x * np.tanh(x_clip))
    names.append('x*tanh(x)')
    features.append(x**2 * np.exp(-x_clip))
    names.append('x^2*exp(-x)')
    
    # Reciprocal (safe)
    features.append(1.0 / (1.0 + np.abs(x)))
    names.append('1/(1+|x|)')
    
    return np.column_stack(features), names

def fit(X_train, y_train):
    start_time = time.time()
    deadline = 300.0
    
    X = X_train
    y = y_train
    n, d = X.shape
    
    # Generate candidate features
    F_all, names = _generate_symbolic_features(X)
    n_features = F_all.shape[1]
    
    # Forward Selection
    # Start with intercept (index 0)
    selected_indices = [0]
    best_mse = np.inf
    current_F = F_all[:, selected_indices]
    
    # Pre-compute y stats for quick MSE
    y_mean = np.mean(y)
    ss_tot = np.sum((y - y_mean)**2)
    if ss_tot == 0:
        ss_tot = 1e-12
    
    # We will fit a linear model with ridge regularization
    reg = 1e-8
    
    # Greedy forward selection
    max_terms = 10
    min_mse_threshold = 1e-10
    
    for step in range(max_terms):
        if time.time() > deadline - 10:
            break
            
        best_candidate_idx = -1
        best_candidate_mse = np.inf
        
        for i in range(n_features):
            if i in selected_indices:
                continue
                
            # Try adding i
            new_indices = selected_indices + [i]
            new_F = F_all[:, new_indices]
            
            try:
                # Solve ridge regression
                A = new_F.T @ new_F + reg * np.eye(len(new_indices))
                b = new_F.T @ y
                w = np.linalg.solve(A, b)
                y_pred = new_F @ w
                mse = np.mean((y_pred - y)**2)
                
                if mse < best_candidate_mse:
                    best_candidate_mse = mse
                    best_candidate_idx = i
                    
            except np.linalg.LinAlgError:
                continue
        
        if best_candidate_idx == -1:
            break
            
        # If improvement is negligible, stop
        if best_candidate_mse >= best_mse * (1 - 1e-6) or best_candidate_mse < min_mse_threshold:
            break
            
        selected_indices.append(best_candidate_idx)
        best_mse = best_candidate_mse
        current_F = F_all[:, selected_indices]
        
        if best_mse < min_mse_threshold:
            break

    # Final fit with selected features
    if len(selected_indices) == 0:
        selected_indices = [0]
        current_F = F_all[:, selected_indices]
        
    try:
        A = current_F.T @ current_F + reg * np.eye(len(selected_indices))
        b = current_F.T @ y
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(current_F, y, rcond=None)[0]
    
    def predict(X_new):
        F_new, _ = _generate_symbolic_features(X_new)
        F_sel = F_new[:, selected_indices]
        return F_sel @ w

    return predict
# EVOLVE-BLOCK-END
