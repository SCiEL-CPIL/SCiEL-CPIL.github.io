"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid: Nelder-Mead from multiple diverse starts for fast early progress on Rosenbrock."""

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
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            fx = float(func(x))
            call_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        dim = self.dim
        budget = self.budget

        # Generate diverse starting points
        # Include the origin (often near the minimum for Rosenbrock variants)
        # and random points in the box
        starts = [np.zeros(dim)]
        for _ in range(19):
            starts.append(rng.uniform(-5, 5, dim))

        # Sort starts by distance to origin (prioritize closer starts for early progress)
        starts.sort(key=lambda s: np.linalg.norm(s))

        # Run Nelder-Mead from each start with limited budget per start
        # Nelder-Mead is derivative-free but very effective on Rosenbrock
        # It explores the valley structure well
        n_starts = len(starts)
        budget_per_start = max(10, (budget - 50) // n_starts)

        for i, x0 in enumerate(starts):
            if not time_ok() or call_count >= budget - 20:
                break

            # Adaptive budget: use more budget for earlier (closer) starts
            if i < 5:
                maxfun = budget_per_start * 2
            elif i < 15:
                maxfun = budget_per_start
            else:
                maxfun = budget_per_start // 2

            try:
                res = minimize(
                    tracked_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'maxfev': maxfun,
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

        # Final phase: CMA-ES-like refinement with (1,1) strategy
        if time_ok() and call_count < budget - 10:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.3
            remaining = budget - call_count - 10

            for _ in range(remaining):
                if not time_ok() or call_count >= budget - 5:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = tracked_func(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.85

        return best_x, best_fx
# EVOLVE-BLOCK-END
