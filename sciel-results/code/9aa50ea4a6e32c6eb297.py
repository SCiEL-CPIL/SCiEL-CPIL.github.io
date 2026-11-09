"""Optimizer for BBOB f14 (different powers) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import time
import numpy as np

from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead + adaptive ES with budget and wall-clock guards."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start = time.time()
        DEADLINE = 325.0  # safety margin before the 330s kill

        rng = np.random.default_rng(12345)
        cache = {}
        calls = 0
        best_x = None
        best_f = np.inf

        def evalf(x):
            nonlocal calls, best_x, best_f
            key = tuple(np.round(x, 9))
            val = cache.get(key)
            if val is None:
                val = float(func(np.asarray(x, dtype=float)))
                cache[key] = val
                calls += 1
            if val < best_f:
                best_f = val
                best_x = np.asarray(x, dtype=float).copy()
            return val

        def budget_ok():
            return calls < budget

        def time_ok():
            return (time.time() - start) < DEADLINE

        # ---- Phase 1: multi-start Nelder-Mead global search ----
        # Initial points: center + a set of spread points.
        starts = [np.zeros(dim)]
        for _ in range(12):
            starts.append(rng.uniform(-4.5, 4.5, dim))

        for s in starts:
            if not (budget_ok() and time_ok()):
                break
            try:
                res = minimize(
                    evalf,
                    np.asarray(s, dtype=float),
                    method="Nelder-Mead",
                    options={
                        "maxiter": 250,
                        "maxfev": 200,
                        "xatol": 1e-6,
                        "fatol": 1e-6,
                    },
                )
            except Exception:
                pass

        # ---- Phase 2: adaptive (1+1)-ES polish from the best point ----
        if budget_ok() and time_ok() and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            for _ in range(budget - calls):
                if not (budget_ok() and time_ok()):
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evalf(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                else:
                    sigma *= 0.87

        # ---- Phase 3: final high-precision polish (tight tolerance) ----
        if budget_ok() and time_ok() and best_x is not None:
            try:
                minimize(
                    evalf,
                    best_x.copy(),
                    method="Nelder-Mead",
                    options={
                        "maxiter": 200,
                        "maxfev": 80,
                        "xatol": 1e-9,
                        "fatol": 1e-9,
                    },
                )
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
        return best_x, best_f
# EVOLVE-BLOCK-END
