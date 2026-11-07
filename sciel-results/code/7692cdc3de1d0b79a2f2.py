import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

N = 50

def compute_value(h):
    h = np.asarray(h)
    if not np.all(np.isfinite(h)):
        return -np.inf
    # Clip to non-negative to handle optimizer noise, though DE bounds this
    # Clipping makes the function well-defined everywhere
    h = np.clip(h, 0.0, None)
    
    if np.max(h) == 0:
        return -np.inf

    g = np.convolve(h, h)
    m = len(g)
    
    # y = [0, g[0], ..., g[m-1], 0]
    y = np.zeros(m + 2)
    y[1:-1] = g
    
    w = 1.0 / (m + 1)
    
    y_curr = y[:m+1]
    y_next = y[1:m+2]
    terms = y_curr**2 + y_curr * y_next + y_next**2
    l2 = (w / 3.0) * np.sum(terms)
    
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    
    if norm_1 == 0 or norm_inf == 0:
        return -np.inf
        
    return l2 / (norm_1 * norm_inf)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Initial guess: uniform
    best_h = np.ones(N)
    best_val = compute_value(best_h)
    
    # Differential Evolution
    bounds = [(0.0, 1.0) for _ in range(N)]
    
    try:
        # Use a seed for reproducibility if desired, but random is fine
        # maxiter and popsize chosen to fit within time
        # 50 dims, pop 15, iter 50 -> 37500 evals
        res = differential_evolution(
            lambda x: -compute_value(x), 
            bounds, 
            maxiter=50, 
            popsize=15, 
            tol=1e-8, 
            seed=42,
            init='latinhypercube'
        )
        
        if not res.success and not res.message.startswith('Function evaluation at x0'):
             # It might have hit maxiter, which is fine
             pass
             
        if res.fun < -best_val: # Since we minimized -value
            best_h = res.x
            best_val = -res.fun
            
    except Exception as e:
        # Fallback if DE fails
        pass
    
    # Check time
    current_time = time.time()
    if current_time < deadline - 20.0:
        # Local Refinement using Powell
        # Powell is good for non-smooth
        try:
            res_local = minimize(
                lambda x: -compute_value(x), 
                best_h, 
                method='Powell', 
                options={'maxiter': 1000, 'maxfev': 5000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if res_local.fun < -best_val:
                best_h = res_local.x
                best_val = -res_local.fun
        except Exception as e:
            pass
            
    # Ensure final result is valid and non-negative
    best_h = np.clip(best_h, 0.0, None)
    
    # If the result is invalid (e.g. all zeros), fall back to seed
    final_val = compute_value(best_h)
    if final_val <= 1.0 or not np.isfinite(final_val):
        return [1.0] * N
        
    return list(best_h)
