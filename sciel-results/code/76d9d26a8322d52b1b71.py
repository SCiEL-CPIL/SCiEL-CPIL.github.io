"""Rewritten optimizer for BBOB f8 (Rosenbrock) with time guard."""
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Adaptive (1+1)-ES with L-BFGS-B polish, time-guarded."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 285.0  # safety margin before 300s limit
        calls = 0
        dim = self.dim
        max_calls = self.budget

        # Wrap func to count calls and clip
        def f(x):
            nonlocal calls
            if calls >= max_calls:
                return 1e10
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            calls += 1
            val = func(x)
            if not np.isfinite(val):
                return 1e10
            return float(val)

        def time_ok():
            return time.time() < deadline

        # Phase 1: Adaptive (1+1)-ES with multiple random restarts
        rng = np.random.default_rng(12345)
        best_x = None
        best_f = np.inf

        # Use a few restarts to explore different regions
        num_restarts = 3
        calls_per_restart = max(1, (max_calls - 150) // num_restarts)  # reserve 150 for polish

        for r in range(num_restarts):
            if not time_ok() or calls >= max_calls - 50:
                break

            x = rng.uniform(-4.0, 4.0, dim)
            fx = f(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.0
            sigma_min = 1e-12
            sigma_max = 10.0

            # Run (1+1)-ES for this restart
            while calls < max_calls - 50 and time_ok():
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = f(y)
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma = min(sigma * 1.25, sigma_max)
                else:
                    sigma = max(sigma * 0.8, sigma_min)

                # Early exit if we're stuck
                if sigma < 1e-8:
                    break

        # Phase 2: L-BFGS-B local polish from best point
        if calls < max_calls and time_ok() and best_x is not None:
            try:
                def lbfgs_func(xv):
                    if not time_ok() or calls >= max_calls:
                        raise TimeoutError()
                    return f(xv)

                res = minimize(
                    lbfgs_func,
                    best_x,
                    method="L-BFGS-B",
                    bounds=[(-5.0, 5.0)] * dim,
                    options={
                        "maxiter": max(1, min(200, (max_calls - calls) // 2)),
                        "ftol": 1e-14,
                        "gtol": 1e-8,
                        "maxls": 50,
                    },
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        # Phase 3: Nelder-Mead if budget remains
        if calls < max_calls and time_ok() and best_x is not None:
            try:
                def nm_func(xv):
                    if not time_ok() or calls >= max_calls:
                        raise TimeoutError()
                    return f(xv)

                res = minimize(
                    nm_func,
                    best_x,
                    method="Nelder-Mead",
                    options={
                        "maxiter": max(1, min(100, (max_calls - calls) // (dim + 1))),
                        "xatol": 1e-8,
                        "fatol": 1e-10,
                        "adaptive": True,
                    },
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        # Phase 4: Final fine-tuning ES steps if budget remains
        if calls < max_calls and time_ok() and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for _ in range(min(50, max_calls - calls)):
                if not time_ok() or calls >= max_calls:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = f(y)
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma = min(sigma * 1.1, 1.0)
                else:
                    sigma = max(sigma * 0.9, 1e-12)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = f(best_x)

        return best_x, best_f
