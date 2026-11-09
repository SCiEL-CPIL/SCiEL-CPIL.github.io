"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 580  # Safety margin before 630s limit
    
    # 12 positive double roots
    # Start with a reasonable initial guess: spread out roots
    # For Laguerre-related constructions, roots often follow patterns
    # Try initial guess: roots at 1, 2, 3, ..., 12 or similar
    
    def objective(x):
        # x is array of 12 positive root locations
        # Need a proxy for C4 upper bound
        # C4 likely relates to some symmetric function of roots
        # Try: minimize the product of (1 + r_i) or similar
        # Or minimize variance to keep roots compact
        # Actually, for uncertainty bounds, we might want to minimize
        # the spread or a specific moment
        
        # Let's try: C4 proxy = sum of r_i^2 (second moment)
        # Or product of r_i
        # Or sum of 1/r_i
        
        # Based on typical uncertainty principles, the bound often
        # involves the product of variances. For roots, a good proxy
        # might be the geometric mean or the spread.
        
        # Try: minimize sum of squared deviations from mean (variance)
        # This keeps roots compact
        mean_r = np.mean(x)
        variance = np.mean((x - mean_r) ** 2)
        
        # Also penalize very small or very large roots
        penalty = np.sum(np.log(x) ** 2)  # penalize extreme log values
        
        return variance + 0.1 * penalty
    
    def objective2(x):
        # Alternative: minimize the product of roots (geometric mean)
        # Smaller product might give better bound
        return np.sum(np.log(x))
    
    def objective3(x):
        # Minimize sum of 1/x (harmonic mean related)
        return np.sum(1.0 / x)
    
    # Try multiple objectives and pick best
    best_result = None
    best_score = float('inf')
    
    initial_guesses = [
        np.linspace(0.5, 12.0, 12),
        np.linspace(1.0, 10.0, 12),
        np.ones(12) * 3.0,
        np.geomspace(1.0, 10.0, 12),
    ]
    
    for init in initial_guesses:
        if time.time() > deadline:
            break
        try:
            result = minimize(objective, init, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-10})
            if result.success and result.fun < best_score:
                best_score = result.fun
                best_result = result.x.copy()
        except:
            pass
    
    if best_result is None:
        # Fallback: return initial guess
        return [float(x) for x in np.linspace(1.0, 10.0, 12)]
    
    # Ensure all roots are positive
    best_result = np.maximum(best_result, 0.01)
    
    return [float(x) for x in best_result]
# EVOLVE-BLOCK-END
