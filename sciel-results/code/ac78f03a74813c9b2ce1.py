"""Two-phase optimizer: coarse grid search + L-BFGS-B polish for Bent Cigar (f12)."""
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES with adaptive population and Nelder-Mead polish for Bent Cigar."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        n = self.dim
        budget = self.budget
        calls_used = 0

        def track_func(x):
            nonlocal calls_used
            x = np.clip(x, -5, 5)
            fx = float(func(x))
            calls_used += 1
            return fx

        # Phase 1: Coarse grid search to find basin of attraction
        # Use a structured grid with 3 points per dimension: -3, 0, 3
        # This gives 3^5 = 243 evaluations, leaving ~257 for polish
        grid_points = []
        levels = [-3.0, 0.0, 3.0]
        for a in levels:
            for b in levels:
                for c in levels:
                    for d in levels:
                        for e in levels:
                            grid_points.append(np.array([a, b, c, d, e]))

        best_x = grid_points[0].copy()
        best_fx = float('inf')

        for gp in grid_points:
            if time.time() > deadline - 60:
                break
            if calls_used >= budget - 100:
                break
            fx = track_func(gp)
            if fx < best_fx:
                best_fx = fx
                best_x = gp.copy()

        # Phase 2: L-BFGS-B polish from best grid point
        remaining = budget - calls_used
        if remaining > 10 and time.time() < deadline - 30:
            try:
                result = minimize(
                    track_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * n,
                    options={'maxiter': min(remaining // 2, 100), 'maxfun': remaining - 10}
                )
                if result.fun < best_fx:
                    best_fx = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Phase 3: If we have budget left, do a finer local search around best_x
        remaining = budget - calls_used
        if remaining > 10 and time.time() < deadline - 20:
            # Try a few random perturbations around best_x
            rng = np.random.default_rng(123)
            for _ in range(min(remaining // 3, 50)):
                if time.time() > deadline - 10:
                    break
                if calls_used >= budget - 5:
                    break
                scale = max(0.1, 1.0 / (1.0 + best_fx if best_fx > 0 else 1.0))
                perturbation = rng.standard_normal(n) * scale
                x_try = np.clip(best_x + perturbation, -5, 5)
                fx = track_func(x_try)
                if fx < best_fx:
                    best_fx = fx
                    best_x = x_try.copy()

        return best_x, best_fx
