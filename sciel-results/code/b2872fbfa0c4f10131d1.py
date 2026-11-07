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

        def counted_func(x):
            nonlocal calls
            calls += 1
            return func(np.clip(x, -5, 5))

        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, d)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Phase 1: (1+1)-ES with 1/5 success rule and restarts
        es_budget = int(budget * 0.75)
        max_es_calls = es_budget - 1

        sigma = 2.0
        successes = 0
        total_steps = 0
        no_improve = 0
        no_improve_limit = 40

        while calls < es_budget and time_ok():
            y = x + sigma * rng.standard_normal(d)
            fy = func(np.clip(y, -5, 5))
            calls += 1
            total_steps += 1

            if fy < fx:
                x = np.clip(y, -5, 5)
                fx = fy
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
                successes += 1
                no_improve = 0
            else:
                no_improve += 1

            # 1/5 success rule: adapt sigma every 10 steps
            if total_steps % 10 == 0:
                rate = successes / 10.0
                if rate > 0.2:
                    sigma *= 1.5
                elif rate < 0.2:
                    sigma /= 1.5
                sigma = np.clip(sigma, 0.01, 5.0)
                successes = 0

            # Restart from best point if stuck
            if no_improve >= no_improve_limit:
                x = best_x + sigma * rng.standard_normal(d)
                fx = func(np.clip(x, -5, 5))
                calls += 1
                no_improve = 0
                sigma = 1.5
                if calls >= es_budget:
                    break

        # Phase 2: Nelder-Mead polish from best point
        if time_ok() and calls < budget:
            try:
                from scipy.optimize import minimize
                polish_budget = budget - calls
                if polish_budget > 5:
                    nm_func_calls = [0]
                    def nm_func(v):
                        nm_func_calls[0] += 1
                        return func(np.clip(v, -5, 5))
                    
                    res = minimize(
                        nm_func,
                        best_x,
                        method='Nelder-Mead',
                        options={'maxiter': polish_budget, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                    )
                    calls += nm_func_calls[0]
                    if res.fun < best_fx:
                        best_x = np.clip(res.x, -5, 5)
                        best_fx = res.fun
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
