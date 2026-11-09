"""Hybrid (1+1)-ES with final Nelder-Mead polish for BBOB f9."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """(1+1)-ES with 1/5 success rule, followed by Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        rng = np.random.default_rng(0)

        # Phase 1: (1+1)-ES with 1/5 success rule
        # Reserve some budget for polishing
        es_budget = int(self.budget * 0.85)
        polish_budget = self.budget - es_budget

        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        sigma = 2.0
        calls_made = 1

        for _ in range(es_budget - 1):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_made += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                sigma *= 0.87

        # Phase 2: Nelder-Mead polish from best point found
        if polish_budget > 5 and time.time() < deadline:
            try:
                nm_calls = [0]
                nm_limit = polish_budget

                def nm_func(x_in):
                    if time.time() > deadline:
                        return np.inf
                    if nm_calls[0] >= nm_limit:
                        return np.inf
                    nm_calls[0] += 1
                    try:
                        return func(x_in)
                    except Exception:
                        return np.inf

                res = minimize(
                    nm_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': nm_limit * 3,
                        'xatol': 1e-6,
                        'fatol': 1e-6,
                        'adaptive': True,
                        'initial_simplex_size': 0.5
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
