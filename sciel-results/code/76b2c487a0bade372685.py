"""Optimizer for BBOB f9 (rotated Rosenbrock) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start L-BFGS-B with a coarse grid to find the narrow basin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        best_x = None
        best_f = np.inf
        calls_used = 0

        def safe_func(x):
            nonlocal calls_used
            calls_used += 1
            # Clip to box
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            if not np.isfinite(val):
                return 1e10
            return val

        from scipy.optimize import minimize

        # Strategy:
        # 1. Coarse grid search to locate the basin (Rosenbrock valley is narrow).
        # 2. Refine with L-BFGS-B from the best grid point and its neighbors.
        # 3. If budget remains, try additional random starts with L-BFGS-B.

        # Step 1: Coarse grid
        # 5 points per dimension -> 3^5 = 243 points. That's well within budget.
        grid_vals = np.linspace(-4.5, 4.5, 3)  # -4.5, 0, 4.5
        grid_points = np.array(np.meshgrid(*([grid_vals] * self.dim))).T.reshape(-1, self.dim)

        # Evaluate all grid points
        grid_results = []
        for i, x in enumerate(grid_points):
            if calls_used >= self.budget - 10:
                break
            fx = safe_func(x)
            grid_results.append((fx, x.copy()))

        if not grid_results:
            # Fallback: just do a simple random search
            x0 = np.random.uniform(-5, 5, self.dim)
            fx = safe_func(x0)
            return x0, fx

        grid_results.sort(key=lambda t: t[0])
        best_f = grid_results[0][0]
        best_x = grid_results[0][1].copy()

        # Step 2: Refine with L-BFGS-B from top grid points
        n_refine = min(5, len(grid_results))
        for i in range(n_refine):
            if calls_used >= self.budget - 50:
                break
            if time.time() > deadline - 10:
                break

            x0 = grid_results[i][1].copy()
            try:
                res = minimize(
                    safe_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * self.dim,
                    options={'maxiter': 200, 'maxfun': self.budget - calls_used - 10}
                )
                if res.success or res.fun < best_f:
                    if np.isfinite(res.fun) and res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5.0, 5.0).copy()
            except Exception:
                pass

            if calls_used >= self.budget - 20:
                break

        # Step 3: Additional random starts if budget remains
        rng = np.random.default_rng(42)
        while calls_used < self.budget - 10 and time.time() < deadline - 5:
            # Sample near the current best or randomly
            if np.random.random() < 0.7 and best_x is not None:
                x0 = np.clip(best_x + rng.normal(0, 1.0, self.dim), -5.0, 5.0)
            else:
                x0 = rng.uniform(-5.0, 5.0, self.dim)

            try:
                res = minimize(
                    safe_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * self.dim,
                    options={'maxiter': 100, 'maxfun': max(10, self.budget - calls_used - 5)}
                )
                if np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5.0, 5.0).copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
