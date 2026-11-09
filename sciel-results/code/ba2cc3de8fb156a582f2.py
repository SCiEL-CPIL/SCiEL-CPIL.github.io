"""Deterministic grid-based global search followed by local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Two-phase optimizer:
      Phase 1 (70% budget): Dense Halton sequence sampling to map the landscape.
      Phase 2 (30% budget): Nelder-Mead local refinement from the best point found.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _halton_sequence(self, count, dim, base=None):
        """Generate a Halton sequence in [0, 1)^dim."""
        if base is None:
            primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
            base = primes[:dim]

        points = []
        for i in range(1, count + 1):
            point = []
            for d, p in enumerate(base):
                f = 1.0
                r = 0.0
                n = i
                while n > 0:
                    f /= p
                    r += f * (n % p)
                    n //= p
                point.append(r)
            points.append(point)
        return np.array(points)

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Phase 1: Global search with Halton sequence (70% of budget)
        # Phase 2: Local refinement with Nelder-Mead (30% of budget)
        n_global = int(budget * 0.7)
        n_local = budget - n_global
        n_local = max(n_local, 10) # Ensure some local search

        best_x = None
        best_f = np.inf
        total_calls = 0

        # Generate Halton points for the global search
        # Map [0, 1)^dim to [-5, 5]^dim
        halton_global = self._halton_sequence(n_global, dim)
        halton_global = halton_global * 10.0 - 5.0

        # Evaluate global points
        for i in range(n_global):
            x = halton_global[i]
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
                'adaptive': True
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
