import numpy as np
from scipy.optimize import minimize
import time

N = 50

def compute_value(h):
    """Compute the evaluator's value for a given set of step heights."""
    h = np.asarray(h, dtype=np.float64)
    if h.shape != (N,) or np.any(h < 0) or not np.all(np.isfinite(h)):
        return -1.0
    g = np.convolve(h, h)
    m = len(g)
    # y has m+2 elements: [0, g[0], ..., g[m-1], 0]
    y = np.zeros(m + 2)
    y[1:m+1] = g
    # spacing w
    w = 1.0 / (m + 1)
    # l2 = sum over i in 0..m of (w/3) * (y[i]^2 + y[i]*y[i+1] + y[i+1]^2)
    # y has indices 0..m+1, we sum i from 0 to m
    yi = y[:m+1]
    yi1 = y[1:m+2]
    l2 = (w / 3.0) * np.sum(yi**2 + yi * yi1 + yi1**2)
    norm_1 = np.sum(np.abs(g)) / (m + 1)
    norm_inf = np.max(np.abs(g))
    if norm_1 == 0 or norm_inf == 0:
        return -1.0
    return l2 / (norm_1 * norm_inf)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill
    
    # Start with the seed
    h0 = np.ones(N)
    best_h = h0.copy()
    best_val = compute_value(h0)
    
    # Define objective (we want to maximize value, so minimize negative)
    def objective(h):
        v = compute_value(h)
        return -v if v > -0.5 else 1.0
    
    # Use L-BFGS-B with bounds to keep h nonnegative
    bounds = [(0.0, None)] * N
    
    # Multiple restarts with random perturbations
    rng = np.random.RandomState(42)
    n_restarts = 200
    
    for i in range(n_restarts):
        if time.time() > deadline:
            break
        
        # Create a perturbed initial guess
        if i == 0:
            h_init = h0.copy()
        else:
            # Random perturbation around current best or random
            if rng.random() < 0.3:
                h_init = np.abs(rng.randn(N)) * 0.5
            else:
                h_init = np.abs(best_h + rng.randn(N) * 0.3)
        
        # Ensure nonnegativity
        h_init = np.maximum(h_init, 0.0)
        
        # Optimize
        try:
            result = minimize(objective, h_init, method='L-BFGS-B', 
                             bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-12})
            if result.success or result.fun < -best_val:
                val = -result.fun
                if val > best_val:
                    best_val = val
                    best_h = result.x.copy()
        except Exception:
            pass
    
    # Final polish with tighter tolerance
    if time.time() < deadline - 5.0:
        try:
            result = minimize(objective, best_h, method='L-BFGS-B',
                             bounds=bounds, options={'maxiter': 5000, 'ftol': 1e-15})
            if result.fun < -best_val:
                best_val = -result.fun
                best_h = result.x.copy()
        except Exception:
            pass
    
    # Ensure nonnegativity and finiteness
    best_h = np.maximum(best_h, 0.0)
    best_h = np.where(np.isfinite(best_h), best_h, 0.0)
    
    return best_h.tolist()
