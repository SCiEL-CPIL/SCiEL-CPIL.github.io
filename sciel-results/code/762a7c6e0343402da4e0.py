"""Model-and-optimize: structured exploration + L-BFGS-B local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Structured exploration with Sobol sequence + L-BFGS-B local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit
        calls = 0
        dim = self.dim
        budget = self.budget

        # Safe wrapper
        def safe_func(x):
            nonlocal calls
            if calls >= budget:
                return np.inf
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            try:
                val = func(x)
                val = float(val)
                if not np.isfinite(val):
                    val = np.inf
            except Exception:
                val = np.inf
            calls += 1
            return val

        # Phase 1: Sobol-based structured exploration
        # Use scipy's Sobol quasi-random sequence for good coverage
        from scipy.stats import qmc

        # Determine how many Sobol points to use for exploration
        # Reserve some budget for local refinement
        n_explore = min(budget // 2, 200)
        n_refine_budget = budget - n_explore

        # Generate Sobol points in [-5, 5]^dim
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_unit = sampler.random(n_explore)
        sobol_points = -5.0 + 10.0 * sobol_unit

        # Evaluate all Sobol points
        best_x = None
        best_f = np.inf
        all_points = []

        for i in range(n_explore):
            if time.time() > deadline:
                break
            if calls >= budget:
                break
            x = sobol_points[i]
            f = safe_func(x)
            all_points.append((x, f))
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Also try a few random points for diversity
        rng = np.random.default_rng(123)
        n_random = min(20, budget - calls)
        for _ in range(n_random):
            if time.time() > deadline:
                break
            if calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            f = safe_func(x)
            all_points.append((x, f))
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 2: Local refinement using L-BFGS-B from best points
        # Sort by function value and refine top candidates
        all_points.sort(key=lambda p: p[1])
        top_k = min(5, len(all_points))

        for i in range(top_k):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            x0 = all_points[i][0]
            f0 = all_points[i][1]

            try:
                result = minimize(
                    safe_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'maxfun': 50}
                )
                if result.fun < best_f and np.isfinite(result.fun):
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

            # Try a smaller neighborhood refinement
            if time.time() > deadline or calls >= budget:
                break
            try:
                result2 = minimize(
                    safe_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'maxfun': 30, 'gtol': 1e-8}
                )
                if result2.fun < best_f and np.isfinite(result2.fun):
                    best_f = result2.fun
                    best_x = result2.x.copy()
            except Exception:
                pass

        # Phase 3: If budget remains, do a final (1+1)-ES from best point
        if calls < budget and time.time() < deadline:
            sigma = 0.5
            x = best_x.copy()
            fx = best_f
            remaining = budget - calls
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                if calls >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85

        return best_x, best_f
# EVOLVE-BLOCK-END
