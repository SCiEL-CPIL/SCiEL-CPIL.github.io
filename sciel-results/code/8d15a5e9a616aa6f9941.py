"""Optimized program for Erdos minimum overlap task."""
import numpy as np
from scipy.optimize import minimize
import time

N = 100

def _objective_and_grad(x):
    """Objective: max of all correlation values. Returns value and gradient."""
    h = x
    # Compute full correlation of h with (1-h)
    # np.correlate(h, 1-h, mode='full') gives 2*N-1 values
    corr = np.correlate(h, 1.0 - h, mode='full')
    val = corr.max()
    
    # Gradient: d/dh_i of max(corr)
    # corr[k] = sum_j h[j] * (1-h[j+k]) for valid j
    # d corr[k] / d h[i] = (1-h[i+k]) if i+k valid, minus h[i-k] if i-k valid... 
    # Actually np.correlate(a,b,'full')[k] = sum_j a[j+k] * b[j] for appropriate ranges
    # Let's just use numerical gradient for reliability, or compute analytically
    
    # For the max, gradient is the gradient of the argmax component
    idx = np.argmax(corr)
    
    # np.correlate(h, 1-h, mode='full')[idx]
    # full correlation: result[k] = sum_i h[i] * (1-h)[i + idx - (N-1)]
    # = sum_i h[i] * (1 - h[i + idx - N + 1])
    # where indices must be valid
    
    offset = idx - (N - 1)  # this is the lag
    # result = sum_i h[i] * (1 - h[i + offset]) for valid i, i+offset
    # d/dh[i] = (1 - h[i+offset]) - h[i-offset] (if i-offset valid)
    
    grad = np.zeros(N)
    for i in range(N):
        g = 0.0
        # contribution from h[i] as first factor: (1 - h[i+offset]) if i+offset in [0,N)
        if 0 <= i + offset < N:
            g += (1.0 - h[i + offset])
        # contribution from h[i] as second factor (shifted): -h[i-offset] if i-offset in [0,N)
        if 0 <= i - offset < N:
            g -= h[i - offset]
        grad[i] = g
    
    return val, grad

def construct():
    deadline = time.time() + 145.0  # 5s safety margin
    
    # Start from seed
    h0 = np.full(N, 0.5)
    
    best_h = h0.copy()
    best_val = None
    
    # Compute initial value
    def compute_value(h):
        corr = np.correlate(h, 1.0 - h, mode='full')
        return corr.max() / N * 2.0
    
    best_val = compute_value(h0)
    
    # Use L-BFGS-B with projection onto simplex-like constraints
    # Constraints: sum(h) = 50, 0 <= h <= 1
    # We'll use the equality constraint and bounds
    
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - N/2.0, 
                    'jac': lambda x: np.ones(N)}]
    bounds = [(0.0, 1.0)] * N
    
    # Run L-BFGS-B optimization
    try:
        result = minimize(
            _objective_and_grad,
            h0,
            method='L-BFGS-B',
            jac=True,
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 2000, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
        )
        
        if time.time() < deadline:
            h_opt = result.x
            # Project to ensure constraints are satisfied
            h_opt = np.clip(h_opt, 0.0, 1.0)
            # Adjust to satisfy sum constraint
            diff = N/2.0 - np.sum(h_opt)
            if abs(diff) > 1e-9:
                h_opt += diff / N
                h_opt = np.clip(h_opt, 0.0, 1.0)
            
            val_opt = compute_value(h_opt)
            if val_opt < best_val:
                best_h = h_opt.copy()
                best_val = val_opt
                
    except Exception:
        pass
    
    if time.time() >= deadline:
        return best_h
    
    # Try multiple random restarts with time remaining
    rng = np.random.default_rng(42)
    
    while time.time() < deadline - 5.0:
        # Generate random starting point
        h_rand = rng.uniform(0, 1, N)
        # Normalize to sum to N/2
        h_rand = h_rand / np.sum(h_rand) * N/2.0
        h_rand = np.clip(h_rand, 0.0, 1.0)
        
        try:
            result = minimize(
                _objective_and_grad,
                h_rand,
                method='L-BFGS-B',
                jac=True,
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50}
            )
            
            if time.time() >= deadline - 5.0:
                break
                
            h_opt = result.x
            h_opt = np.clip(h_opt, 0.0, 1.0)
            diff = N/2.0 - np.sum(h_opt)
            if abs(diff) > 1e-9:
                h_opt += diff / N
                h_opt = np.clip(h_opt, 0.0, 1.0)
            
            val_opt = compute_value(h_opt)
            if val_opt < best_val:
                best_h = h_opt.copy()
                best_val = val_opt
                
        except Exception:
            continue
    
    return best_h
