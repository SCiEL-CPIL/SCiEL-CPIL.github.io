"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 290.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        best_x = rng.uniform(-5, 5, dim)
        best_f = func(best_x)
        n_calls = 1

        # Multi-start strategy: try several random initial points
        # Each local optimization uses a limited number of function evaluations
        n_starts = 10
        calls_per_start = max(10, (budget - n_calls) // n_starts)

        for i in range(n_starts):
            if time.time() > deadline:
                break
            if n_calls >= budget:
                break

            # Generate a random starting point
            x0 = rng.uniform(-4, 4, dim)
            
            # Run L-BFGS-B from this start
            # We need to count function calls inside the optimizer
            # Since we can't easily hook into scipy's call counter, 
            # we'll use a wrapper or just rely on the total budget.
            # Actually, we can use a closure to count calls.
            
            calls_in_this = [0]
            def wrapped_func(x):
                nonlocal n_calls
                if n_calls >= budget or time.time() > deadline:
                    return best_f # Return a large value to stop
                fx = func(np.clip(x, -5, 5))
                n_calls += 1
                calls_in_this[0] += 1
                return fx

            try:
                result = minimize(wrapped_func, x0, method='L-BFGS-B', 
                                  bounds=[(-5, 5)] * dim,
                                  options={'maxiter': 100, 'maxfun': calls_per_start})
                if result.success and result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
