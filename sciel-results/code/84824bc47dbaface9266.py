"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Hybrid (1+1)-ES with restarts for global exploration, then Nelder-Mead for local polishing."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s limit

        rng = np.random.default_rng(0)

        # Global best
        x_best = rng.uniform(-5, 5, self.dim)
        fx_best = func(x_best)

        calls_made = 1
        remaining = self.budget - 1

        # Parameters for (1+1)-ES with restarts
        sigma = 1.5
        sigma_min = 1e-3
        sigma_max = 8.0
        stall_limit = 25  # Restart after this many unsuccessful steps
        stall_count = 0

        # Track best for potential Nelder-Mead polish
        x_nm_best = x_best.copy()
        fx_nm_best = fx_best

        # Phase 1: (1+1)-ES with restarts for global exploration
        # Use about 70% of budget for exploration
        explore_budget = int(0.7 * self.budget)
        explore_calls = 0

        while (remaining > 0 and time.time() < deadline and explore_calls < explore_budget):
            y = np.clip(x_best + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_made += 1
            remaining -= 1
            explore_calls += 1

            if fy < fx_best:
                x_best, fx_best = y, fy
                sigma = min(sigma * 1.08, sigma_max)
                stall_count = 0
                if fx_best < fx_nm_best:
                    x_nm_best, fx_nm_best = x_best.copy(), fx_best
            else:
                sigma = max(sigma * 0.92, sigma_min)
                stall_count += 1

                if stall_count >= stall_limit:
                    # Restart from a new random point
                    stall_count = 0
                    sigma = 1.5
                    x_new = rng.uniform(-5, 5, self.dim)
                    fx_new = func(x_new)
                    calls_made += 1
                    remaining -= 1
                    explore_calls += 1
                    if fx_new < fx_best:
                        x_best, fx_best = x_new, fx_new
                        if fx_best < fx_nm_best:
                            x_nm_best, fx_nm_best = x_best.copy(), fx_best
                    else:
                        # Keep current best, just reset sigma
                        pass

        # Phase 2: Nelder-Mead for local polishing on the best point found
        # Use remaining budget for Nelder-Mead iterations
        if remaining > 5 and time.time() < deadline:
            # Try Nelder-Mead from the best point
            try:
                def wrapped_func(x):
                    nonlocal calls_made, remaining
                    x_clipped = np.clip(x, -5, 5)
                    fx = func(x_clipped)
                    calls_made += 1
                    remaining -= 1
                    return fx

                # Nelder-Mead with adaptive settings
                result = minimize(
                    wrapped_func,
                    x_nm_best,
                    method='Nelder-Mead',
                    options={
                        'maxiter': min(remaining, 200),
                        'maxfev': remaining,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True,
                        'initial_simplex': None
                    }
                )

                if result.fun < fx_nm_best:
                    x_nm_best = np.clip(result.x, -5, 5)
                    fx_nm_best = result.fun
            except Exception:
                pass

            # Try Nelder-Mead from a few other good points if budget remains
            if remaining > 20 and time.time() < deadline:
                # Generate a few candidate starting points near the best
                for _ in range(min(3, remaining // 20)):
                    if remaining <= 10 or time.time() >= deadline:
                        break
                    x_try = np.clip(x_nm_best + 0.5 * rng.standard_normal(self.dim), -5, 5)
                    try:
                        def wrapped_func2(x):
                            nonlocal calls_made, remaining
                            x_clipped = np.clip(x, -5, 5)
                            fx = func(x_clipped)
                            calls_made += 1
                            remaining -= 1
                            return fx

                        result2 = minimize(
                            wrapped_func2,
                            x_try,
                            method='Nelder-Mead',
                            options={
                                'maxiter': min(remaining, 100),
                                'maxfev': remaining,
                                'xatol': 1e-8,
                                'fatol': 1e-10,
                                'adaptive': True
                            }
                        )

                        if result2.fun < fx_nm_best:
                            x_nm_best = np.clip(result2.x, -5, 5)
                            fx_nm_best = result2.fun
                    except Exception:
                        pass

        # Final check: use remaining budget for a few more random evaluations if needed
        while remaining > 0 and time.time() < deadline:
            x_rand = np.clip(x_nm_best + 0.3 * rng.standard_normal(self.dim), -5, 5)
            fx_rand = func(x_rand)
            calls_made += 1
            remaining -= 1
            if fx_rand < fx_nm_best:
                x_nm_best, fx_nm_best = x_rand, fx_rand

        return x_nm_best, fx_nm_best
# EVOLVE-BLOCK-END
