"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start L-BFGS-B optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(7)
        n = self.dim
        budget = self.budget
        bounds = [(-5, 5)] * n

        # Count function calls
        call_count = [0]

        def wrapped_func(x):
            call_count[0] += 1
            return func(x)

        x_best = None
        f_best = np.inf

        # Use a fraction of budget for random restarts, then refine
        # Start with several random points
        n_restarts = min(10, max(3, budget // 50))
        remaining_budget = budget

        for i in range(n_restarts):
            if remaining_budget < 10:
                break
                
            x0 = rng.uniform(-4, 4, n)
            f0 = wrapped_func(x0)
            remaining_budget -= 1
            if f0 < f_best:
                f_best = f0
                x_best = x0.copy()
            
            # Try L-BFGS-B from this point
            if remaining_budget < 5:
                continue
                
            try:
                res = minimize(
                    wrapped_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': max(1, remaining_budget // 2),
                             'maxls': 10,
                             'gtol': 1e-5}
                )
                remaining_budget -= (call_count[0] - (budget - remaining_budget))
                
                if res.success or res.fun < f_best:
                    if res.fun < f_best:
                        f_best = res.fun
                        x_best = res.x.copy()
            except Exception:
                pass
            
            if remaining_budget <= 0:
                break

        # Final polish with a few more local optimizations from best point
        if remaining_budget > 5 and x_best is not None:
            try:
                res = minimize(
                    wrapped_func,
                    x_best,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': max(1, remaining_budget // 2),
                             'maxls': 10,
                             'gtol': 1e-6}
                )
                if res.fun < f_best:
                    f_best = res.fun
                    x_best = res.x.copy()
            except Exception:
                pass

        return x_best, f_best
# EVOLVE-BLOCK-END
