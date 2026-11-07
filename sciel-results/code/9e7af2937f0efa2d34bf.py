"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid: Nelder-Mead for basin finding, L-BFGS-B for polishing."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 40s safety margin before 330s kill

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        best_x = np.zeros(self.dim)
        best_fx = np.inf
        call_count = 0

        def tracked_func(x):
            nonlocal call_count, best_x, best_fx
            # The problem statement says func clips, but we should also ensure
            # our internal tracking doesn't go out of bounds if we use it for logic.
            # However, we just pass to func.
            x_arr = np.asarray(x, dtype=float)
            fx = float(func(x_arr))
            call_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x_arr.copy()
            return fx

        # Phase 1: Nelder-Mead from 3 diverse starts
        # Nelder-Mead is derivative-free and often fast in low dimensions for smooth functions.
        # We limit the budget per start to ensure we explore multiple basins.
        n_nm_starts = 3
        starts = [
            np.zeros(self.dim),
            rng.uniform(-2, 2, self.dim),
            rng.uniform(-2, 2, self.dim)
        ]

        nm_budget_per_start = 30

        for x0 in starts:
            if not time_ok() or call_count >= self.budget - 100:
                break
            try:
                res = minimize(
                    tracked_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': nm_budget_per_start,
                        'xatol': 1e-5,
                        'fatol': 1e-5
                    }
                )
            except Exception:
                pass

        # Phase 2: L-BFGS-B from the best point found
        # L-BFGS-B is a gradient-based method that converges very quickly for smooth functions
        # once it is in the basin of attraction.
        if time_ok() and call_count < self.budget - 10:
            x_start = best_x.copy()
            try:
                res = minimize(
                    tracked_func,
                    x_start,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 200,
                        'maxfun': self.budget - call_count - 5,
                        'ftol': 1e-12
                    }
                )
            except Exception:
                pass

        # Phase 3: (1+1)-ES polish if budget remains
        # This handles any remaining noise or if L-BFGS-B stopped early due to convergence criteria
        # but didn't reach the true minimum.
        if time_ok() and call_count < self.budget - 5:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.1
            stuck = 0
            remaining = self.budget - call_count - 5

            for _ in range(remaining):
                if not time_ok() or call_count >= self.budget - 5:
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = tracked_func(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.5
                    stuck = 0
                else:
                    sigma *= 0.87
                    stuck += 1

                if stuck > 20:
                    x = np.clip(best_x + 0.2 * rng.standard_normal(self.dim), -5, 5)
                    fx = tracked_func(x)
                    sigma = 0.05
                    stuck = 0

        return best_x, best_fx
# EVOLVE-BLOCK-END
