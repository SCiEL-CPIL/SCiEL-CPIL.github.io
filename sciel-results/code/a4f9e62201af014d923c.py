"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, time guard, and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 285  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        sigma = 2.0
        calls = 1

        # Phase 1: Exploration with (1+1)-ES
        # Reserve ~10% of budget for polish phase
        es_budget = int(self.budget * 0.9)
        while calls < es_budget:
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
            else:
                sigma *= 0.87

        # Phase 2: Final polish pass with Nelder-Mead
        if calls < self.budget:
            remaining = self.budget - calls
            if remaining > 10:
                try:
                    from scipy.optimize import minimize
                    call_count = [0]
                    def wrapped_func(z):
                        nonlocal fx
                        call_count[0] += 1
                        if call_count[0] > remaining:
                            return fx
                        return func(np.clip(z, -5, 5))

                    # Start from current best, use tight tolerances
                    res = minimize(wrapped_func, x, method='Nelder-Mead',
                                   options={'maxiter': 200, 'xatol': 1e-10,
                                            'fatol': 1e-12, 'adaptive': True})
                    if res.fun < fx:
                        x, fx = np.clip(res.x, -5, 5), res.fun
                except Exception:
                    pass

        return x, fx
# EVOLVE-BLOCK-END
