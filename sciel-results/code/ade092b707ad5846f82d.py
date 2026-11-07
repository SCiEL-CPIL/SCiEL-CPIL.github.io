"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start local optimizer using SciPy for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        # Multi-start strategy
        # We will use 5 different starting points.
        # Each run is limited to budget // 5 evaluations.
        n_starts = 5
        evals_per_start = max(10, budget // n_starts)
        
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf

        for i in range(n_starts):
            if time.time() > deadline:
                break
            
            # Random start in [-4, 4]
            x0 = rng.uniform(-4, 4, dim)
            
            # Use Nelder-Mead method
            try:
                res = minimize(func, x0, method='Nelder-Mead', 
                               options={'maxfev': evals_per_start, 'xatol': 1e-6, 'fatol': 1e-6})
                # Update best
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # If no result was found (e.g. all exceptions), return a random point
        if best_x is None:
            best_x = rng.uniform(-4, 4, dim)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
