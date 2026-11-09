"""Deterministic grid-based global search followed by local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """
    Two-phase optimizer:
      Phase 1 (80% budget): Global search using Sobol sequence to map the landscape.
      Phase 2 (20% budget): Nelder-Mead local refinement from the best point found.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Split budget: 80% global, 20% local refinement
        n_global = int(budget * 0.8)
        n_local = budget - n_global
        
        # Ensure minimum calls for stability
        n_global = max(n_global, 10)
        n_local = max(n_local, 10)
        
        # Adjust if sum exceeds budget due to max constraints
        if n_global + n_local > budget:
            n_local = budget - n_global
            if n_local < 10:
                n_global = budget - 10
                n_local = 10

        best_x = None
        best_f = np.inf
        total_calls = 0

        # Phase 1: Global Search with Sobol Sequence
        # Sobol sequences have better uniformity properties than Halton for low-discrepancy sampling
        try:
            sobol = qmc.Sobol(d=dim, scramble=True)
            points = sobol.random(n_global)
            # Map from [0, 1)^dim to [-5, 5]^dim
            points = points * 10.0 - 5.0
        except Exception:
            # Fallback to Halton if Sobol fails (unlikely)
            points = np.random.rand(n_global, dim) * 10.0 - 5.0

        for i in range(n_global):
            x = points[i]
            fx = func(x)
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Local refinement using Nelder-Mead
        if n_local > 0 and best_x is not None and total_calls < budget:
            calls_made = [0]
            max_calls = min(n_local, budget - total_calls)

            def wrapped_func(x):
                if calls_made[0] >= max_calls:
                    return np.inf
                calls_made[0] += 1
                return func(np.clip(x, -5, 5))

            options = {
                'maxiter': max_calls,
                'maxfev': max_calls,
                'xatol': 1e-8,
                'fatol': 1e-10,
            }

            try:
                result = minimize(
                    wrapped_func,
                    best_x,
                    method='Nelder-Mead',
                    options=options
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
