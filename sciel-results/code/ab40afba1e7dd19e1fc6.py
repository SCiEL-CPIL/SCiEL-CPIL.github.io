"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, restarts, and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        d = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, d)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Phase 1: (1+1)-ES with 1/5 success rule and restarts
        es_budget = int(budget * 0.75)

        sigma = 2.0
        sigma_min = 0.05
        sigma_max = 4.0
        no_improve = 0
        no_improve_limit = 40
        success_count = 0
        window_size = 5

        for _ in range(es_budget - 1):
            if not time_ok():
                break
            if calls >= es_budget:
                break

            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
                no_improve = 0
                success_count += 1
                # 1/5 success rule: increase sigma if success rate > 1/5
                if success_count > window_size / 5:
                    sigma = min(sigma * 1.25, sigma_max)
            else:
                no_improve += 1
                # 1/5 success rule: decrease sigma if success rate <= 1/5
                if success_count <= window_size / 5:
                    sigma = max(sigma * 0.85, sigma_min)

            # Reset window every window_size steps
            if calls % window_size == 0:
                success_count = 0

            # Restart from best point if stuck
            if no_improve >= no_improve_limit:
                if calls >= es_budget:
                    break
                x = best_x + sigma * rng.standard_normal(d)
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls += 1
                no_improve = 0
                sigma = min(2.0, sigma_max)
                success_count = 0

        # Phase 2: Nelder-Mead polish from best point
        if time_ok():
            try:
                from scipy.optimize import minimize

                # Track calls made by Nelder-Mead
                nm_calls = [0]

                def nm_func(v):
                    if nm_calls[0] >= budget - calls:
                        return best_fx
                    nm_calls[0] += 1
                    return func(np.clip(v, -5, 5))

                polish_budget = budget - calls
                if polish_budget > 10:
                    res = minimize(
                        nm_func,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': polish_budget, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                    )
                    calls += nm_calls[0]
                    if res.fun < best_fx:
                        best_x = np.clip(res.x, -5, 5)
                        best_fx = res.fun
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
