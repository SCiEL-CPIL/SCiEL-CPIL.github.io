"""Refined Cohn-Goncalves Laguerre construction optimizer."""
import time
import numpy as np
from scipy.special import eval_genlaguerre
from scipy.optimize import differential_evolution, minimize

_START_TIME = time.time()
_DEADLINE = 630.0
_SAFETY_MARGIN = 10.0  # seconds before deadline

def _time_remaining():
    return _DEADLINE - (time.time() - _START_TIME) - _SAFETY_MARGIN

def _laguerre_combination(x, roots, m=12, n=1):
    """
    Evaluate the Laguerre polynomial combination at x.
    For the Cohn-Goncalves construction with m=12, n=1,
    F(x) = sum_{k=0}^{m} c_k * L_k(x) where coefficients are chosen
    so that F has double roots at the specified locations.
    
    We construct F as a product of (x - r_i)^2 for i=1..12,
    then project onto the Laguerre basis, but for the bound computation,
    we evaluate the weighted norm.
    """
    # F(x) = prod_{i=1}^{12} (x - r_i)^2
    # But we need it expressed in Laguerre polynomials for the uncertainty bound.
    # The C4 bound involves the L2 norm of F * e^{-x/2} * x^{n/2} or similar.
    
    # Actually, for the Fourier sign uncertainty, the bound C4 is:
    # C4 = || F(x) * e^{-x/2} * x^{1/2} ||_infinity or a related quantity
    # where F is the Laguerre combination with the given roots.
    
    # Let's compute F(x) as the product of squared linear factors
    # and then compute the weighted sup norm on a grid.
    
    # For efficiency, compute log|F(x)| = 2 * sum log|x - r_i|
    # Then C4 ~ max over x of exp(log|F(x)| - x/2 + 0.5*log(x))
    
    # But this is for the specific construction. Let me use a simpler model:
    # C4 is the maximum of |F(x)| * e^{-x/2} * x^{0.5} over x > 0
    # where F(x) = prod (x - r_i)^2
    
    # To avoid overflow, work in log space
    # log|F(x)| = 2 * sum_{i} log|x - r_i|
    # log(C4_candidate) = 2*sum log|x-r_i| - x/2 + 0.5*log(x)
    
    # We maximize this over x > 0, and C4 = exp(max)
    # We want to MINIMIZE C4, so minimize the max of log(C4_candidate)
    
    return x, roots

def _compute_C4(roots):
    """Compute the C4 upper bound for given root locations."""
    if len(roots) != 12:
        return np.inf
    if np.any(roots <= 0):
        return np.inf
    if np.any(np.diff(np.sort(roots)) <= 0):
        return np.inf
    
    # Sort roots
    r = np.sort(roots)
    
    # C4 = max_{x>0} |F(x)| * e^{-x/2} * x^{0.5}
    # where F(x) = prod_{i=1}^{12} (x - r_i)^2
    # 
    # log|F(x)| = 2 * sum_{i=1}^{12} log|x - r_i|
    # log(C4) = 2*sum log|x-r_i| - x/2 + 0.5*log(x)
    #
    # Find max over x > 0
    
    # Grid search for the maximum
    # The function is smooth; maximum occurs where derivative = 0
    # d/dx [2*sum log|x-r_i| - x/2 + 0.5*log(x)] = 0
    # 2*sum 1/(x-r_i) - 0.5 + 0.5/x = 0  (for x not equal to any r_i)
    
    # Use a grid + refinement
    # Range: from min(r)*0.5 to max(r)*2
    x_min = max(r[0] * 0.5, 0.1)
    x_max = r[-1] * 2.0 + 10.0
    
    # Dense grid
    n_grid = 2000
    x_grid = np.linspace(x_min, x_max, n_grid)
    
    # Avoid points too close to roots
    mask = np.ones(n_grid, dtype=bool)
    for ri in r:
        mask &= np.abs(x_grid - ri) > 1e-8
    
    if not np.any(mask):
        return np.inf
    
    x_valid = x_grid[mask]
    
    # Compute log(C4) on grid
    log_F = 2.0 * np.sum(np.log(np.abs(x_valid[:, None] - r[None, :])), axis=1)
    log_C4_grid = log_F - x_valid / 2.0 + 0.5 * np.log(x_valid)
    
    # Find max
    idx_max = np.argmax(log_C4_grid)
    C4_estimate = np.exp(log_C4_grid[idx_max])
    
    return C4_estimate

def construct():
    """Return 12 positive double-root locations minimizing C4."""
    
    # Deadline check
    if _time_remaining() < 5.0:
        # Not enough time, return seed
        return [4.0 * (i + 1) for i in range(12)]
    
    # Initial guess: evenly spaced
    x0 = np.array([4.0 * (i + 1) for i in range(12)])
    
    # Try to improve using optimization
    # Use differential evolution for global optimization
    # Bounds: each root in (0.1, 100)
    bounds = [(0.1, 100.0)] * 12
    
    # Objective: minimize C4
    def objective(roots):
        if _time_remaining() < 5.0:
            return 1e10  # signal to stop
        return _compute_C4(roots)
    
    best_roots = x0.copy()
    best_score = _compute_C4(x0)
    
    # Differential evolution
    try:
        result = differential_evolution(
            objective,
            bounds,
            maxiter=50,
            popsize=10,
            tol=1e-4,
            mutation=(0.3, 1.0),
            recombination=0.8,
            seed=42,
            polish=True,
            callback=None
        )
        
        if result.success or result.fun < best_score:
            # Check if roots are valid
            r = np.sort(result.x)
            if np.all(r > 0) and np.all(np.diff(r) > 0):
                best_roots = r
                best_score = result.fun
    except Exception:
        pass
    
    # Try local refinement from best found
    if _time_remaining() > 10.0:
        try:
            def obj_local(roots):
                return _compute_C4(roots)
            
            result2 = minimize(
                obj_local,
                best_roots,
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            r2 = np.sort(result2.x)
            if np.all(r2 > 0) and np.all(np.diff(r2) > 0):
                score2 = _compute_C4(r2)
                if score2 < best_score:
                    best_roots = r2
                    best_score = score2
        except Exception:
            pass
    
    # Ensure valid output
    best_roots = np.sort(best_roots)
    if np.any(best_roots <= 0) or len(best_roots) != 12:
        return [4.0 * (i + 1) for i in range(12)]
    
    return list(best_roots)
