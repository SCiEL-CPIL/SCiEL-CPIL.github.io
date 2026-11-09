import numpy as np
import time
import math
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 315

        best_x = np.zeros(self.dim)
        best_f = float('inf')
        eval_count = 0
        rng = np.random.default_rng(0)

        def try_eval(x):
            nonlocal eval_count, best_f, best_x
            if eval_count >= self.budget:
                return None
            if time.time() > deadline:
                return None
            try:
                fx = func(np.asarray(x, dtype=float))
            except Exception:
                return None
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = np.asarray(x, dtype=float).copy()
            return fx

        def halton_sequence(n, dim, skip=10):
            """Generate Halton quasi-random sequence in [0,1]^dim."""
            points = np.zeros((n, dim))
            for d in range(dim):
                base = d + 2  # prime bases: 2, 3, 5, 7, 11
                denom = 1.0
                for i in range(n):
                    f = base + i + skip
                    num = 1.0
                    while f > 0:
                        f, r = divmod(f, base)
                        num = (num + r / base) / base
                    points[i, d] = num
            return points

        # Phase 1: Global exploration with Halton quasi-random sampling
        n_halton = min(40, self.budget // 10)
        halton_pts = halton_sequence(n_halton, self.dim)
        for i in range(n_halton):
            if eval_count >= self.budget or time.time() > deadline:
                break
            x = halton_pts[i] * 10.0 - 5.0
            try_eval(x)

        # Phase 2: Multi-start Nelder-Mead local optimization
        # Select diverse starting points from the best found so far
        # We'll do multiple NM restarts from different regions

        # Create candidate starting points: best point + points near it + spread
        nm_starts = []

        # Always start from the best
        nm_starts.append(best_x.copy())

        # Add points at various scales around the best
        scales = [1.0, 2.0, 3.0, 1.5, 2.5, 0.5, 0.8, 1.2]
        for s in scales:
            x_start = best_x + s * rng.standard_normal(self.dim)
            x_start = np.clip(x_start, -5.0, 5.0)
            nm_starts.append(x_start)

        # Add a few more diverse random starts
        for _ in range(6):
            x_start = rng.uniform(-4.5, 4.5, self.dim)
            nm_starts.append(x_start)

        # Shuffle starts to diversify
        perm = rng.permutation(len(nm_starts))
        nm_starts = [nm_starts[i] for i in perm]

        # Run Nelder-Mead from each start, with budget allocation
        # Each NM run gets a portion of remaining budget
        remaining = self.budget - eval_count
        n_starts = len(nm_starts)
        budget_per_start = max(10, remaining // n_starts)

        for start in nm_starts:
            if eval_count >= self.budget or time.time() > deadline - 5:
                break

            # Local budget for this NM run
            local_budget = min(budget_per_start, self.budget - eval_count)
            if local_budget < 5:
                break

            # Nelder-Mead with custom callback to track evaluations
            nm_eval_count = [0]
            nm_max_evals = local_budget

            def nm_func(x, _count=nm_eval_count):
                if _count[0] >= nm_max_evals:
                    return best_f + 1e10
                fx = try_eval(x)
                _count[0] += 1
                if fx is None:
                    return best_f + 1e10
                return fx

            try:
                result = minimize(
                    nm_func,
                    start,
                    method='Nelder-Mead',
                    options={
                        'maxiter': nm_max_evals,
                        'maxfev': nm_max_evals,
                        'xatol': 1e-6,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
            except Exception:
                pass

            # Update best from NM result
            if result.success or result.fun < best_f:
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = np.asarray(result.x, dtype=float).copy()

        # Phase 3: Fine local polish with (1+1)-ES
        remaining = self.budget - eval_count
        if remaining > 5 and time.time() < deadline - 3:
            rng_local = np.random.default_rng(42)
            x = best_x.copy()
            fx = best_f
            sigma = 0.02
            no_improve = 0
            limit = 20

            while remaining > 0 and time.time() < deadline - 3:
                if no_improve > limit:
                    sigma = 0.05
                    no_improve = 0

                y = np.clip(x + sigma * rng_local.standard_normal(self.dim), -5, 5)
                fy = try_eval(y)
                remaining -= 1

                if fy is None:
                    break

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.05
                    no_improve = 0
                else:
                    sigma *= 0.95
                    no_improve += 1
                    if sigma < 1e-6:
                        break

        return best_x, best_f
