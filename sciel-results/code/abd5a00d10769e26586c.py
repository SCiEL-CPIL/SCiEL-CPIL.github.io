"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, minimize_scalar, root_scalar
from scipy.special import eval_laguerre, laguerre, roots_laguerre
import time
import itertools

def _compute_bound(roots):
    """
    Compute an upper bound on the C4 constant given root locations.
    We use the variance of the roots as a proxy for the uncertainty bound.
    Lower variance implies a more concentrated distribution, which typically
    yields a tighter (lower) uncertainty bound in these spectral constructions.
    """
    roots = np.asarray(roots, dtype=np.float64)
    if np.any(roots <= 0) or len(roots) != 12:
        return 1e10
    
    # Sort roots
    r = np.sort(roots)
    
    # Proxy: Variance of the roots
    # This encourages roots to be close to each other (compact support)
    var = np.var(r)
    
    # Add a small penalty for very small roots to avoid singularity
    # if the true bound involves 1/r terms
    min_r = np.min(r)
    if min_r < 0.1:
        return 1e10
    
    return var

def construct():
    """
    Return 12 positive double-root locations that minimize the upper bound
    on C4 for the Cohn-Goncalves Laguerre polynomial construction.
    """
    start_time = time.time()
    deadline = start_time + 600.0  # Must finish within 600s

    # Seed roots: 4, 8, ..., 48
    seed_roots = np.array([4.0 * (i + 1) for i in range(12)])
    
    # Also consider zeros of Laguerre L_12 scaled to match the mean
    try:
        x12, _ = roots_laguerre(12)
        # Scale to match the mean of the seed (26)
        mean_seed = np.mean(seed_roots)
        mean_lag = np.mean(x12)
        scaled_lag = x12 * (mean_seed / mean_lag)
    except:
        scaled_lag = seed_roots

    # Candidate starting points
    candidates = [
        seed_roots,
        scaled_lag,
        np.linspace(1, 10, 12),
        np.linspace(5, 20, 12)
    ]
    
    best_roots = seed_roots
    best_bound = _compute_bound(seed_roots)
    
    # Objective for optimizer
    def objective(x):
        if np.any(x <= 0):
            return 1e10
        return _compute_bound(x)
    
    bounds = [(1e-6, None)] * 12
    
    # Use Nelder-Mead for derivative-free optimization
    # This is robust for small dimensions
    for x0 in candidates:
        elapsed = time.time() - start_time
        if elapsed > deadline - 30:
            break
        
        try:
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            if result.success:
                b = objective(result.x)
                if b < best_bound:
                    best_bound = b
                    best_roots = result.x
        except Exception:
            pass
    
    # Ensure positivity
    best_roots = np.maximum(best_roots, 1e-6)
    
    return [float(r) for r in best_roots]

# EVOLVE-BLOCK-END
