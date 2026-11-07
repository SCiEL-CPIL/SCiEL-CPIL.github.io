"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start adaptive restart (1+1)-ES with Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 330s

        # Track global best
        best_x = None
        best_f = np.inf
        calls = 0

        def track(func_call):
            nonlocal calls
            calls += 1
            return func_call

        # Nelder-Mead local polish
        def polish(x0, max_calls=30):
            nonlocal calls, best_x, best_f
            if calls >= budget - 5:
                return x0, func(x0)

            def wrapped(x):
                nonlocal calls
                if calls >= budget - 2:
                    return 1e10
                calls += 1
                return func(np.clip(x, -5, 5))

            try:
                res = minimize(wrapped, x0, method='Nelder-Mead',
                               options={'maxiter': max_calls, 'xatol': 1e-8, 'fatol': 1e-10})
                fx = res.fun
                if fx < best_f:
                    best_x = res.x.copy()
                    best_f = fx
                return res.x, fx
            except Exception:
                return x0, func(x0)

        # (1+1)-ES with 1/5 success rule
        def run_es(x_init, sigma_init, max_calls):
            nonlocal calls, best_x, best_f
            x = x_init.copy()
            fx = func(x)
            if fx < best_f:
                best_x = x.copy()
                best_f = fx
            calls += 1

            sigma = sigma_init
            successes = 0
            total = 0

            while calls < budget - 5 and time.time() < deadline:
                y = np.clip(x + sigma * np.random.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1

                total += 1
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    if fx < best_f:
                        best_x = x.copy()
                        best_f = fx
                    # 1/5 success rule: increase sigma
                    sigma *= 1.0 + 0.33 * (1.0 / 5.0)  # gentle increase
                else:
                    sigma *= 1.0 - 0.33 * (4.0 / 5.0)  # decrease

                # Stagnation check: if no improvement in last 20 calls
                if total % 20 == 0:
                    if successes / max(total, 1) < 0.1:
                        # Stagnated, trigger restart
                        return x, fx, True

            return x, fx, False

        # Multi-start with adaptive restarts
        np.random.seed(42)
        n_restarts = 0
        max_restarts = 10

        # Initial start
        x0 = np.random.uniform(-5, 5, dim)
        x, fx, stagnated = run_es(x0, 2.0, budget)

        # Adaptive restarts
        while n_restarts < max_restarts and calls < budget - 10 and time.time() < deadline:
            if not stagnated:
                break

            # Try Nelder-Mead polish on current best
            if best_x is not None and calls < budget - 15:
                polish_x, polish_f = polish(best_x, max_calls=20)
                n_restarts += 1
                continue

            # Restart from a new random point, but biased toward best
            if best_x is not None:
                x0 = best_x + np.random.normal(0, 1.5, dim)
                x0 = np.clip(x0, -5, 5)
            else:
                x0 = np.random.uniform(-5, 5, dim)

            x, fx, stagnated = run_es(x0, 1.5, budget)
            n_restarts += 1

        # Final polish if we have budget left
        if best_x is not None and calls < budget - 5 and time.time() < deadline:
            polish(best_x, max_calls=min(20, budget - calls - 2))

        return best_x, best_f
# EVOLVE-BLOCK-END
