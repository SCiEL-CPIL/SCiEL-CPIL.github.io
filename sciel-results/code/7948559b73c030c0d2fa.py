"""
SLSQP-based local search with multi-start for BBOB f7 (Step ellipsoidal) in 5D.

Strategy: Use scipy.optimize.minimize with SLSQP (which handles bounds natively)
from multiple random starting points. For the step-ellipsoidal function, the
landscape is piecewise smooth, so a gradient-based method can quickly converge
to a local minimum once in the right basin. Multi-start helps cover different
basins. We allocate most of the budget to the best few starts.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer using multi-start SLSQP local search.
    
    For the step-ellipsoidal function f7, the landscape consists of flat regions
    (steps) connected by steep drops. SLSQP can efficiently navigate the smooth
    regions and converge to local minima. Multiple random starts increase the
    chance of finding the global minimum.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        # Track best solution
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        # Count function calls
        def count_func(x):
            nonlocal calls_made
            if calls_made >= budget:
                return np.inf
            calls_made += 1
            val = func(x)
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Bounds for SLSQP
        bounds = [(-5.0, 5.0)] * dim

        # Multi-start: generate several random starting points
        # We'll use a mix of center and random points
        rng = np.random.RandomState(42)
        
        # Phase 1: Explore with several random starts
        # Use about 60% of budget for exploration
        explore_budget = int(budget * 0.6)
        
        # Number of random starts - keep it modest to allow convergence
        n_starts = 8
        
        for s in range(n_starts):
            if time.time() > deadline or calls_made >= budget:
                break
            
            if s == 0:
                # First start at center
                x0 = np.zeros(dim)
            else:
                # Random starts in [-4, 4]^5 (since min is in this region)
                x0 = rng.uniform(-4.0, 4.0, dim)
            
            # Check if we've already tried this point
            if np.allclose(x0, best_x):
                continue
            
            # Run SLSQP with limited budget
            # Each local search should use a reasonable number of evaluations
            # SLSQP in 5D typically converges in < 50 evaluations
            try:
                res = minimize(count_func, x0, method='SLSQP', bounds=bounds,
                             options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            except Exception:
                # If SLSQP fails, try Nelder-Mead as fallback
                try:
                    res = minimize(count_func, x0, method='Nelder-Mead', bounds=bounds,
                                 options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-10})
                except Exception:
                    continue
            
            if time.time() > deadline or calls_made >= budget:
                break
        
        # Phase 2: Refine the best solution found
        # Use remaining budget for a final local refinement
        if calls_made < budget and best_f < np.inf:
            try:
                # Use the best found as starting point for refinement
                res = minimize(count_func, best_x, method='SLSQP', bounds=bounds,
                             options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-10})
            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
