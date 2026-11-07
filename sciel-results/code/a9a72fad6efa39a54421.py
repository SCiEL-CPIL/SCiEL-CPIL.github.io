"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, plus Nelder-Mead polish."""

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
        sigma = 2.0
        calls = 1

        # Main (1+1)-ES phase: use ~85% of budget
        es_budget = int(budget * 0.85)
        for _ in range(es_budget - 1):
            if not time_ok():
                break
            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
            else:
                sigma *= 0.85

        # Nelder-Mead polish phase: use remaining budget
        if time_ok():
            try:
                from scipy.optimize import minimize
                # Use remaining budget for polish
                polish_budget = budget - calls
                if polish_budget > 10:
                    res = minimize(
                        lambda v: func(np.clip(v, -5, 5)),
                        x,
                        method='Nelder-Mead',
                        options={'maxiter': polish_budget, 'xatol': 1e-10, 'fatol': 1e-12}
                    )
                    if res.fun < fx:
                        x = np.clip(res.x, -5, 5)
                        fx = res.fun
            except Exception:
                pass

        return x, fx
# EVOLVE-BLOCK-END
