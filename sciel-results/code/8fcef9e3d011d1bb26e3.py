"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Hybrid: random sampling + Nelder-Mead with restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Phase 1: Random sampling to find a good starting point
        # Use 30% of budget for random search
        n_random = max(10, budget // 3)
        n_random = min(n_random, budget - 1)

        best_x = None
        best_f = np.inf

        for _ in range(n_random):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Nelder-Mead local optimization with restarts
        remaining = budget - n_random
        if remaining <= 0:
            return best_x, best_f

        # Split remaining budget among several NM runs
        n_restarts = 5
        budget_per_restart = max(1, remaining // n_restarts)

        # Initial simplex size
        simplex_size = 1.0

        for restart in range(n_restarts):
            if best_f <= 1e-10:
                break

            # For the first restart, use the best found point
            # For subsequent restarts, try points near the best or random perturbations
            if restart == 0:
                x0 = best_x.copy()
            elif restart == 1:
                x0 = best_x + simplex_size * rng.standard_normal(dim)
                x0 = np.clip(x0, -5, 5)
            else:
                # Mix of local perturbation and global random
                if rng.random() < 0.5:
                    x0 = best_x + simplex_size * rng.standard_normal(dim)
                    x0 = np.clip(x0, -5, 5)
                else:
                    x0 = rng.uniform(-5, 5, dim)

            # Run Nelder-Mead
            def objective(x):
                return func(np.clip(x, -5, 5))

            try:
                result = minimize(
                    objective,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': budget_per_restart,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5, 5)
            except Exception:
                pass

            # Shrink simplex size for subsequent restarts
            simplex_size *= 0.5

        return best_x, best_f
# EVOLVE-BLOCK-END
