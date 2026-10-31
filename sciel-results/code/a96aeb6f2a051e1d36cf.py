"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

N = 100

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _compute_overlap(h, shift):
    """Compute overlap of h with shifted 1-h for given circular shift."""
    shifted_h = np.roll(h, shift)
    # Overlap: sum of h_i * (1 - shifted_h_i)
    return np.sum(h * (1 - shifted_h))

def _max_overlap(h):
    """Maximum overlap over all shifts."""
    overlaps = [_compute_overlap(h, s) for s in range(N)]
    return max(overlaps)

def construct():
    deadline = time.time() + 100  # Safety margin before 150s limit
    
    # Start with uniform
    h0 = np.full(N, 0.5)
    best_h = h0.copy()
    best_val = _max_overlap(h0)
    
    # Try gradient-based optimization on a smooth surrogate
    # Use soft-max to make the objective smooth
    def objective(h_flat):
        h = h_flat
        overlaps = np.array([_compute_overlap(h, s) for s in range(N)])
        # Soft-max approximation
        beta = 50.0
        soft_max = np.logsumexp(beta * overlaps) / beta
        return soft_max
    
    # Constraints: sum(h) = N/2, 0 <= h_i <= 1
    constraints = [{'type': 'eq', 'fun': lambda h: np.sum(h) - N/2}]
    bounds = [(0, 1)] * N
    
    # Try from uniform start
    try:
        res = minimize(objective, h0, method='SLSQP', bounds=bounds,
                      constraints=constraints, options={'maxiter': 500, 'ftol': 1e-10})
        if res.success and np.sum(np.abs(res.x - 0.5)) > 1e-6:
            val = _max_overlap(res.x)
            if val < best_val:
                best_h = res.x.copy()
                best_val = val
    except Exception:
        pass
    
    if time.time() > deadline - 20:
        return best_h
    
    # Try from a few random initializations
    np.random.seed(42)
    for trial in range(10):
        if time.time() > deadline - 10:
            break
        
        # Random initialization with correct sum
        h_init = np.random.uniform(0, 1, N)
        h_init = (h_init - h_init.mean()) + 0.5
        h_init = np.clip(h_init, 0, 1)
        # Adjust to make sum exactly N/2
        diff = N/2 - np.sum(h_init)
        h_init += diff / N
        h_init = np.clip(h_init, 0, 1)
        # Final correction
        h_init += (N/2 - np.sum(h_init)) / N
        
        try:
            res = minimize(objective, h_init, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 300, 'ftol': 1e-10})
            if res.success:
                val = _max_overlap(res.x)
                if val < best_val:
                    best_h = res.x.copy()
                    best_val = val
        except Exception:
            pass
    
    # Polish: local refinement
    if time.time() < deadline - 5:
        try:
            res = minimize(objective, best_h, method='SLSQP', bounds=bounds,
                          constraints=constraints, options={'maxiter': 500, 'ftol': 1e-12})
            if res.success:
                val = _max_overlap(res.x)
                if val < best_val:
                    best_h = res.x.copy()
                    best_val = val
        except Exception:
            pass
    
    # Ensure exact sum
    best_h = best_h + (N/2 - np.sum(best_h)) / N
    best_h = np.clip(best_h, 0, 1)
    
    return best_h
# EVOLVE-BLOCK-END
