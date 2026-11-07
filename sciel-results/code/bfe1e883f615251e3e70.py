"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        self.dim = self.dim
        budget = self.budget
        dim = self.dim

        # Track global best
        global_best_x = None
        global_best_f = np.inf

        # Sub-budget for each Nelder-Mead run
        # Nelder-Mead is efficient, 50 evaluations is usually enough to converge locally
        sub_budget = 50
        num_runs = max(1, budget // sub_budget)
        
        # We will track calls manually to be safe, but scipy's maxfev should handle it.
        # However, scipy's maxfev might not be 100% strict in all versions or edge cases.
        # We'll use a wrapper to be absolutely sure.
        
        call_count = 0
        def wrapped_func(x):
            nonlocal call_count
            nonlocal global_best_f
            nonlocal global_best_x
            call_count += 1
            fx = func(x)
            if fx < global_best_f:
                global_best_f = fx
                global_best_x = x.copy()
            return fx

        for run in range(num_runs):
            # Determine starting point
            if run == 0:
                x0 = np.zeros(dim)
            else:
                # Random start in the feasible region, slightly inset to avoid boundary issues
                x0 = rng.uniform(-4.5, 4.5, dim)

            # Check if we have enough budget left
            if call_count + sub_budget > budget:
                # Adjust sub_budget for the last run
                current_sub_budget = budget - call_count
                if current_sub_budget < 10: # Too little to be useful
                    break
            else:
                current_sub_budget = sub_budget

            # Run Nelder-Mead
            try:
                res = minimize(wrapped_func, x0, method='Nelder-Mead', 
                               options={'maxfev': current_sub_budget, 'xatol': 1e-5, 'fatol': 1e-5})
            except Exception:
                # If scipy fails for some reason, break
                break

            # Update global best if the result is better (should already be updated by wrapper, 
            # but just in case the wrapper didn't catch the final point if it was better than previous)
            if res.fun < global_best_f:
                global_best_f = res.fun
                global_best_x = res.x.copy()

            # If we hit the max budget, stop
            if call_count >= budget:
                break

        # If no calls were made (should not happen), evaluate one point
        if global_best_x is None:
            x0 = np.zeros(dim)
            global_best_f = func(x0)
            global_best_x = x0.copy()

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
