"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools
from scipy.optimize import minimize


class Optimizer:
    """Grid search + Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill
        n = self.dim
        budget = self.budget
        calls = 0

        def safe_call(x):
            nonlocal calls
            if calls >= budget:
                return float('inf')
            calls += 1
            return func(x)

        # Phase 1: Coarse grid search
        # Minimizer is in [-4, 4]^5. Use levels -4, 0, 4.
        levels = [-4.0, 0.0, 4.0]
        grid_points = list(itertools.product(levels, repeat=n))
        
        # To be safe with time, we might not evaluate all if it's slow, but 243 is fast.
        # We store (f, x) for sorting
        results = []
        for pt in grid_points:
            if calls >= budget or time.time() > deadline:
                break
            x = np.array(pt)
            f = safe_call(x)
            results.append((f, x))
            
        if not results:
            # Fallback
            x0 = np.zeros(n)
            f0 = safe_call(x0)
            best_x = x0
            best_f = f0
            return best_x, best_f

        # Sort by function value
        results.sort(key=lambda item: item[0])
        
        # Get top candidates for local search
        # We have budget - calls remaining.
        # Nelder-Mead in 5D takes ~50-100 evals.
        # Let's try up to 3 restarts.
        top_k = min(3, len(results))
        candidates = [results[i][1] for i in range(top_k)]
        
        best_x = candidates[0]
        best_f = results[0][0]
        
        # Phase 2: Local refinement with Nelder-Mead
        # We will attempt to refine the best points found in Phase 1.
        for i in range(top_k):
            if calls >= budget or time.time() > deadline:
                break
            
            x0 = candidates[i]
            # If we already have a better solution from a previous restart, 
            # we might skip if x0 is worse than current best? 
            # No, NM might find a better local min.
            
            # Calculate max function evaluations allowed for this run
            remaining = budget - calls
            if remaining < 10:
                break
                
            # Nelder-Mead options
            # maxfev is the maximum number of function evaluations
            options = {
                'maxfev': remaining,
                'xatol': 1e-8,
                'fatol': 1e-8
            }
            
            try:
                res = minimize(safe_call, x0, method='Nelder-Mead', options=options)
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                # If minimization fails, continue with next candidate
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
