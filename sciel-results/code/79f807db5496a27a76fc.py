"""New optimizer for BBOB f23 (Katsuura) using Multi-Start Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-Start Nelder-Mead for Katsuura."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        B = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        calls = 0

        n_starts = 5
        budget_per_start = B // n_starts
        # Estimate max iterations to not exceed budget
        # Each NM iteration uses ~d+1 evaluations
        max_iter_per_start = max(1, budget_per_start // (d + 1))

        for k in range(n_starts):
            if calls >= B:
                break
            
            x0 = rng.uniform(-4, 4, size=d)
            
            local_calls = 0
            max_local_calls = budget_per_start
            
            def objective(x):
                nonlocal calls, local_calls
                if calls >= B:
                    return 1e10
                fx = func(x)
                calls += 1
                local_calls += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx
            
            try:
                res = minimize(
                    objective,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter_per_start,
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'adaptive': True
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
