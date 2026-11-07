"""Multi-start Nelder-Mead with coordinate-descent polish for BBOB f22."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Multi-start Nelder-Mead optimizer with coordinate-descent polish for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        def eval_and_track(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            calls += 1
            try:
                val = func(np.asarray(x, dtype=float))
                if not np.isfinite(val):
                    val = best_f
                if val < best_f:
                    best_f = val
                    best_x = np.asarray(x, dtype=float).copy()
                return val
            except Exception:
                return best_f

        rng = np.random.default_rng(12345)

        def golden_section_1d(x, axis, lo, hi, max_iter=12):
            """Optimize a single coordinate using golden-section search."""
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return x

            gr = (1.0 + math.sqrt(5.0)) / 2.0
            inv_gr = 1.0 / gr
            inv_gr2 = 1.0 / (gr * gr)

            x1 = x.copy()
            x1[axis] = lo + inv_gr2 * (hi - lo)
            x2 = x.copy()
            x2[axis] = lo + inv_gr * (hi - lo)

            f1 = eval_and_track(x1)
            f2 = eval_and_track(x2)

            for _ in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    break
                if f1 < f2:
                    hi = x2[axis]
                    x2 = x1.copy()
                    f2 = f1
                    x1 = x.copy()
                    x1[axis] = lo + inv_gr2 * (hi - lo)
                    f1 = eval_and_track(x1)
                else:
                    lo = x1[axis]
                    x1 = x2.copy()
                    f1 = f2
                    x2 = x.copy()
                    x2[axis] = lo + inv_gr * (hi - lo)
                    f2 = eval_and_track(x2)

            if f1 < f2:
                return x1
            else:
                return x2

        def coordinate_descent_polish(x, n_sweeps=2, search_radius=0.3, max_iter_per_dim=8):
            """Lightweight coordinate descent polish for final precision."""
            nonlocal calls, best_f, best_x
            for sweep in range(n_sweeps):
                if calls >= budget or time.time() > deadline:
                    break
                order = rng.permutation(dim)
                for axis in order:
                    if calls >= budget or time.time() > deadline:
                        break
                    lo = max(-5.0, x[axis] - search_radius)
                    hi = min(5.0, x[axis] + search_radius)
                    x = golden_section_1d(x, axis, lo, hi, max_iter=max_iter_per_dim)
            return x

        # Phase 1: Coarse random search to identify promising basins
        n_random = max(15, budget // 3)
        candidates = []

        for i in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, size=dim)
            f_val = eval_and_track(x)
            candidates.append((f_val, x.copy()))

        # Sort by fitness
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Nelder-Mead local refinement on top candidates
        top_k = min(6, len(candidates))

        for i in range(top_k):
            if calls >= budget or time.time() > deadline:
                break

            x0 = candidates[i][1].copy()
            remaining = budget - calls
            if remaining < 20:
                break

            # Allocate a reasonable chunk of remaining budget to each NM run
            nm_budget = max(30, remaining // (top_k - i))
            nm_budget = min(nm_budget, remaining)

            try:
                from scipy.optimize import minimize

                res = minimize(
                    eval_and_track,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max(50, nm_budget),
                        'maxfev': nm_budget,
                        'xatol': 1e-7,
                        'fatol': 1e-9,
                        'adaptive': True,
                    },
                    bounds=[(-5, 5)] * dim,
                )
            except Exception:
                pass

        # Phase 3: Coordinate descent polish around the best point
        if calls < budget and time.time() < deadline:
            x = best_x.copy()
            remaining = budget - calls

            # First polish with moderate radius
            if remaining > 10:
                x = coordinate_descent_polish(x, n_sweeps=2, search_radius=0.5, max_iter_per_dim=8)
                f_val = eval_and_track(x)
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()

            # Second polish with tighter radius
            if calls < budget and time.time() < deadline:
                x = best_x.copy()
                x = coordinate_descent_polish(x, n_sweeps=2, search_radius=0.15, max_iter_per_dim=6)
                f_val = eval_and_track(x)
                if f_val < best_f:
                    best_f = f_val
                    best_x = x.copy()

        # Phase 4: If budget remains, do random restarts from best point
        if calls < budget and time.time() < deadline:
            step = 0.8
            while calls < budget and time.time() < deadline:
                x = best_x + rng.normal(0, step, size=dim)
                x = np.clip(x, -5, 5)
                f_val = eval_and_track(x)
                if f_val < best_f:
                    step = max(0.05, step * 0.85)
                else:
                    step = min(2.5, step * 1.15)

        return best_x, best_f
# EVOLVE-BLOCK-END
