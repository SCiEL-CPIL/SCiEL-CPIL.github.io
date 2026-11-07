"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _objective(h):
    h = np.clip(h, 0, 1)
    # Adjust to satisfy sum constraint
    s = h.sum()
    target = N / 2.0
    h = h + (target - s) / N
    h = np.clip(h, 0, 1)
    corr = np.correlate(h, 1 - h, mode="full")
    return corr.max()

def construct():
    start = time.time()
    deadline = start + 120.0
    
    best_h = np.full(N, 0.5)
    best_val = _objective(best_h)
    
    # Try multiple random initializations and local optimization
    rng = np.random.RandomState(42)
    
    for trial in range(200):
        if time.time() > deadline - 5.0:
            break
        
        # Random initialization
        h0 = rng.uniform(0, 1, N)
        # Enforce sum constraint approximately
        h0 = h0 * (N / 2.0) / h0.sum()
        h0 = np.clip(h0, 0, 1)
        
        try:
            result = minimize(
                _objective, h0,
                method='L-BFGS-B',
                bounds=[(0, 1)] * N,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            h_opt = np.clip(result.x, 0, 1)
            # Enforce sum constraint
            s = h_opt.sum()
            h_opt = h_opt + (N / 2.0 - s) / N
            h_opt = np.clip(h_opt, 0, 1)
            
            val = _objective(h_opt)
            if val < best_val:
                best_val = val
                best_h = h_opt.copy()
        except Exception:
            continue
    
    # Final adjustment for sum constraint
    s = best_h.sum()
    best_h = best_h + (N / 2.0 - s) / N
    best_h = np.clip(best_h, 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END
