"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import scipy.optimize
import time

class Optimizer:
    """Coordinate Descent with 1-D minimization along axes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        evals = 0

        # 1. Initialization: Sample a few points to find a good starting location
        # Sharp ridge has a narrow basin, so random sampling might miss it.
        # However, the problem states the minimizer is in [-4, 4]^5.
        # A small grid or random samples should hit the basin.
        init_budget = max(10, self.budget // 10)
        for _ in range(init_budget):
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            evals += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            if evals >= self.budget:
                break

        if best_x is None:
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            evals += 1

        # 2. Coordinate Descent Loop
        # We use scipy's bounded scalar minimization for each dimension.
        # This effectively performs a line search along each axis.
        
        # We wrap the function to count evaluations accurately
        # because minimize_scalar does not expose the exact number of function calls
        # in a standardized way across all versions, and we must strictly respect the budget.
        
        while evals < self.budget:
            # Time guard: 300s limit, stop at 280s to be safe
            if time.time() - start_time > 280:
                break

            for d in range(self.dim):
                if evals >= self.budget:
                    break

                x_current = best_x.copy()
                
                # Local counter for this 1-D optimization
                local_evals = 0
                
                def func_1d(t):
                    nonlocal local_evals
                    x_current[d] = t
                    val = func(x_current)
                    local_evals += 1
                    return val

                try:
                    # Use bounded method to stay within [-5, 5]
                    # xatol controls the precision
                    res = scipy.optimize.minimize_scalar(
                        func_1d, 
                        bounds=(-5.0, 5.0), 
                        method='bounded',
                        options={'xatol': 1e-4}
                    )
                    
                    # Update total evals
                    evals += local_evals
                    
                    if res.success and res.fun < best_f:
                        best_f = res.fun
                        best_x[d] = res.x
                        
                except Exception:
                    # If an exception occurs, we still consumed some evals
                    evals += local_evals

        return best_x, best_f
# EVOLVE-BLOCK-END
