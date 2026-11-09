"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import differential_evolution, minimize


class Optimizer:
    """Multi-start L-BFGS-B optimizer with adaptive local search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        best_x = np.zeros(self.dim)
        best_f = float('inf')
        calls_made = 0

        def tracked_func(x):
            nonlocal calls_made
            if calls_made >= self.budget:
                return 0.0  # Dummy value to avoid extra calls
            val = func(x)
            calls_made += 1
            return val

        # Phase 1: Differential evolution for global exploration
        # Use a small number of iterations to conserve budget
        de_budget = 300
        de_maxiter = 30
        de_popsize = 10

        def de_objective(x):
            return tracked_func(x)

        try:
            de_result = differential_evolution(
                de_objective,
                bounds=[(-5, 5)] * self.dim,
                maxiter=de_maxiter,
                popsize=de_popsize,
                tol=1e-6,
                seed=42,
                polish=False,
                init='sobol'
            )
        except Exception:
            de_result = None

        if de_result is not None and calls_made < self.budget:
            if de_result.fun < best_f:
                best_f = de_result.fun
                best_x = de_result.x.copy()

        # Phase 2: Local refinement with L-BFGS-B from multiple starting points
        # Use the best point from DE and a few random points
        start_points = [best_x.copy()]
        # Add a few random points near the best
        for _ in range(3):
            x = best_x + rng.normal(0, 0.5, self.dim)
            x = np.clip(x, -5, 5)
            start_points.append(x)

        # Use remaining budget for local optimization
        remaining_budget = self.budget - calls_made
        if remaining_budget > 0:
            # L-BFGS-B with tight tolerance
            try:
                local_result = minimize(
                    tracked_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={'maxiter': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if local_result.fun < best_f:
                    best_f = local_result.fun
                    best_x = local_result.x.copy()
            except Exception:
                pass

        # Phase 3: If budget remains, do a final round of local refinement from multiple starts
        if calls_made < self.budget - 10:
            # Try a few more local optimizations from different starts
            for start in start_points[1:]:
                if calls_made >= self.budget - 10:
                    break
                try:
                    local_result = minimize(
                        tracked_func,
                        start,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * self.dim,
                        options={'maxiter': 20, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    if local_result.fun < best_f:
                        best_f = local_result.fun
                        best_x = local_result.x.copy()
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
