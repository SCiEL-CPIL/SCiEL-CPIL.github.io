"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.special import roots_laguerre

def _compute_c4_bound(roots):
    """
    Compute the upper bound on C4 given 12 double-root locations.
    
    For the Cohn-Goncalves Laguerre construction with m=12, n=1,
    the uncertainty bound involves the product of distances from roots
    and normalization factors.
    
    The bound C4 is related to the reciprocal of the product of the roots
    raised to some power, combined with exponential decay terms.
    """
    roots = np.asarray(roots)
    if np.any(roots <= 0):
        return 1e10
    
    log_prod = np.sum(np.log(roots))
    sum_roots = np.sum(roots)
    
    log_bound = -log_prod - 0.5 * sum_roots
    
    return np.exp(log_bound)

def construct():
    """
    Construct the 12 positive double-root locations for the Laguerre combination (m=12, n=1).
    
    Uses constrained optimization to minimize the Cohn-Goncalves uncertainty bound.
    """
    zeros, _ = roots_laguerre(12)
    zeros = np.sort(zeros)
    
    best_result = None
    best_score = np.inf
    
    n_starts = 20
    rng = np.random.RandomState(42)
    
    for start in range(n_starts):
        if start == 0:
            x0 = zeros.copy()
        else:
            perturbation = rng.uniform(0.1, 0.5, size=12)
            x0 = zeros + perturbation * rng.choice([-1, 1], size=12)
            x0 = np.maximum(x0, 0.1)
        
        def objective(x):
            if np.any(x <= 0):
                return 1e10
            return _compute_c4_bound(x)
        
        constraints = [{'type': 'ineq', 'fun': lambda x: np.min(x) - 0.01}]
        
        result = minimize(
            objective, 
            x0, 
            method='SLSQP',
            constraints=constraints,
            bounds=[(0.01, 100)] * 12,
            options={'maxiter': 1000, 'ftol': 1e-12}
        )
        
        if result.success and result.fun < best_score:
            best_score = result.fun
            best_result = result.x
    
    if best_result is None:
        return [float(z) for z in zeros]
    
    best_roots = np.sort(best_result)
    return [float(r) for r in best_roots]
# EVOLVE-BLOCK-END
