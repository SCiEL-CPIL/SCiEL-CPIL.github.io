"""Optimizer for BBOB f14 (different powers) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import time
import numpy as np

from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """Sobol multi-start Nelder-Mead with adaptive budget allocation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start = time.time()
        DEADLINE = 320.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        cache = {}
        calls = 0
        best_x = None
        best_f = np.inf

        def evalf(x):
            nonlocal calls, best_x, best_f
            x = np.asarray(x, dtype=float)
            key = tuple(np.round(x, 10))
            val = cache.get(key)
            if val is None:
                val = float(func(x))
                cache[key] = val
                calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        def budget_ok():
            return calls < budget

        def time_ok():
            return (time.time() - start) < DEADLINE

        # ---- Phase 1: Sobol-based multi-start Nelder-Mead ----
        # Generate space-filling starting points using Sobol sequence
        n_starts = 16
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        # Use enough points for good coverage
        n_sobol = 32
        sobol_points = sobol.random(n_sobol)
        # Map to [-4.5, 4.5]^dim (slightly inside bounds)
        sobol_mapped = sobol_points * 9.0 - 4.5

        # Include origin and a few random points for diversity
        starts = [np.zeros(dim)]
        starts.extend(sobol_mapped[:n_starts])
        # Add a few random points for extra diversity
        for _ in range(4):
            starts.append(rng.uniform(-4.5, 4.5, dim))

        # Allocate budget: ~60% for global search, ~30% for polish, ~10% reserve
        global_budget = int(budget * 0.6)
        polish_budget = int(budget * 0.3)
        reserve_budget = budget - global_budget - polish_budget

        # Sort starts by distance from center (prefer closer points first)
        starts_sorted = sorted(starts, key=lambda x: np.linalg.norm(x))

        for s in starts_sorted:
            if not (budget_ok() and time_ok()):
                break
            if calls >= global_budget:
                break
            remaining = global_budget - calls
            maxfev = min(remaining, 150)
            if maxfev < 10:
                break
            try:
                res = minimize(
                    evalf,
                    np.clip(s, -4.9, 4.9),
                    method="Nelder-Mead",
                    options={
                        "maxiter": min(200, maxfev),
                        "maxfev": maxfev,
                        "xatol": 1e-7,
                        "fatol": 1e-7,
                    },
                )
            except Exception:
                pass

        # ---- Phase 2: Adaptive (1+1)-ES polish from the best point ----
        if budget_ok() and time_ok() and best_x is not None and calls < budget - polish_budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.3
            max_polish_steps = min(polish_budget, budget - calls)
            for _ in range(max_polish_steps):
                if not (budget_ok() and time_ok()):
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evalf(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.25
                else:
                    sigma *= 0.85

        # ---- Phase 3: Final high-precision polish ----
        if budget_ok() and time_ok() and best_x is not None:
            remaining = budget - calls
            if remaining >= 20:
                try:
                    minimize(
                        evalf,
                        best_x.copy(),
                        method="Nelder-Mead",
                        options={
                            "maxiter": min(150, remaining),
                            "maxfev": remaining,
                            "xatol": 1e-10,
                            "fatol": 1e-10,
                        },
                    )
                except Exception:
                    pass

        if best_x is None:
            best_x = np.zeros(dim)
        return best_x, best_f
# EVOLVE-BLOCK-END
