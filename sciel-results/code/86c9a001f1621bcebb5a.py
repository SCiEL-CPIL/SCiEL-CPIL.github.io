"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead with adaptive restarts for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 15s safety margin before 300s limit
        n = self.dim
        budget = self.budget
        calls = 0

        def _counted_func(x):
            nonlocal calls
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            return fx

        best_x = np.zeros(n)
        best_f = np.inf

        # Phase 1: Sparse deterministic sampling to find a good starting basin
        # Use a 2^5 grid over [-4, 4] (where minimizer lies) = 32 evaluations
        grid_levels = np.array([-4.0, 4.0])
        import itertools
        for combo in itertools.product(*[grid_levels] * n):
            if time.time() > deadline or calls >= budget:
                break
            x = np.array(combo, dtype=float)
            fx = _counted_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Add center point
        if calls < budget and time.time() < deadline:
            x0 = np.zeros(n)
            fx0 = _counted_func(x0)
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

        # Phase 2: Multi-start Powell with restarts
        rng = np.random.default_rng(123)
        num_restarts = 10
        remaining = budget - calls

        for restart in range(num_restarts):
            if calls >= budget or time.time() > deadline:
                break

            # Generate starting point: mix of best-known and random
            if restart == 0:
                x_start = best_x.copy()
            elif restart < 5:
                # Perturb best
                x_start = best_x + rng.normal(scale=1.0, size=n)
                x_start = np.clip(x_start, -5.0, 5.0)
            else:
                # Random restart
                x_start = rng.uniform(-4.5, 4.5, size=n)

            # Set up Powell with tight iteration limit
            max_iter = 50
            maxfev = min(50, budget - calls)
            try:
                result = minimize(
                    _counted_func,
                    x_start,
                    method='Powell',
                    options={
                        'maxiter': max_iter,
                        'maxfev': maxfev,
                        'xtol': 1e-6,
                        'ftol': 1e-8
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.clip(result.x, -5.0, 5.0).copy()
            except Exception:
                pass

            # Check if we've made progress
            if calls >= budget or time.time() > deadline:
                break

        # Phase 3: Final polish with small-step coordinate descent
        if calls < budget - 5 and time.time() < deadline:
            step = 0.05
            for _ in range(50):
                if calls >= budget or time.time() > deadline:
                    break
                improved = False
                for d in range(n):
                    if calls >= budget or time.time() > deadline:
                        break
                    for sign in [1.0, -1.0]:
                        if calls >= budget or time.time() > deadline:
                            break
                        x_p = best_x.copy()
                        x_p[d] += sign * step
                        x_p = np.clip(x_p, -5.0, 5.0)
                        fx_p = _counted_func(x_p)
                        if fx_p < best_f:
                            best_x = x_p.copy()
                            best_f = fx_p
                            improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-6:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
