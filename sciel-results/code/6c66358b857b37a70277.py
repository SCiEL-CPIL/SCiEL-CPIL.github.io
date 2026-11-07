"""CMA-ES optimizer for BBOB f11 (Discus) with adaptive restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        total_budget = self.budget
        n = self.dim
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def wrapper(x):
            nonlocal calls_used
            if calls_used >= total_budget or time.time() > deadline:
                raise StopIteration()
            calls_used += 1
            try:
                return float(func(x))
            except Exception:
                return np.inf

        # Phase 1: Multi-start Nelder-Mead
        # 4 starts: Origin + 3 random
        starts = [np.zeros(n)]
        for _ in range(3):
            starts.append(rng.uniform(-4, 4, n))
        
        # ~50 calls per start
        for x0 in starts:
            if calls_used >= total_budget or time.time() > deadline:
                break
            try:
                res = minimize(
                    wrapper, 
                    x0, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': 10, 
                        'xatol': 1e-3,
                        'fatol': 1e-3
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except (StopIteration, Exception):
                pass

        # Phase 2: Polish with Nelder-Mead from best point
        if best_x is not None and calls_used < total_budget and time.time() < deadline:
            try:
                res = minimize(
                    wrapper, 
                    best_x, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'xatol': 1e-8,
                        'fatol': 1e-8
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except (StopIteration, Exception):
                pass

        if best_x is None:
            best_x = np.zeros(n)
            if calls_used < total_budget:
                calls_used += 1
                try:
                    best_f = float(func(best_x))
                except Exception:
                    best_f = np.inf

        return best_x, best_f
# EVOLVE-BLOCK-END
