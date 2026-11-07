"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-start gradient-based optimizer with local refinement for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        bounds = [(-5.0, 5.0)] * dim

        # Time guard: return best result if approaching deadline
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s

        # Call counter
        calls = 0

        def counting_func(x):
            nonlocal calls
            calls += 1
            # Clip to box
            x_clipped = np.clip(x, -5.0, 5.0)
            return func(x_clipped)

        # Best solution tracking
        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Phase 1: Generate diverse starting points
        # Use a grid of points covering the space, plus some random points
        rng = np.random.default_rng(42)

        # Corner points and midpoints
        start_points = []
        # Corners
        for sign in [-1, 1]:
            for i in range(dim):
                pt = np.zeros(dim)
                pt[i] = sign * 4.0
                start_points.append(pt)
        # Center
        start_points.append(np.zeros(dim))
        # Midpoints along each axis
        for i in range(dim):
            for sign in [-1, 1]:
                pt = np.zeros(dim)
                pt[i] = sign * 2.0
                start_points.append(pt)

        # Add random points
        n_random = max(5, budget // 20)
        for _ in range(n_random):
            start_points.append(rng.uniform(-4.0, 4.0, dim))

        # Phase 2: Run L-BFGS-B from each starting point
        for x0 in start_points:
            if calls >= budget - 10:
                break
            if time.time() > deadline:
                break

            # Clip starting point
            x0_clipped = np.clip(x0, -5.0, 5.0)

            try:
                # Use L-BFGS-B with bounds
                result = minimize(
                    counting_func,
                    x0_clipped,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if result.success or result.fun < best_f:
                    update_best(result.x, result.fun)
            except Exception:
                pass

            # Check budget
            if calls >= budget:
                break

        # Phase 3: Local refinement using Nelder-Mead or Powell from best point
        if best_x is not None and calls < budget - 5:
            if time.time() < deadline:
                try:
                    result = minimize(
                        counting_func,
                        best_x,
                        method='Nelder-Mead',
                        bounds=bounds,
                        options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                    )
                    if result.fun < best_f:
                        update_best(result.x, result.fun)
                except Exception:
                    pass

        # Phase 4: If we have budget left, do a final local search with tighter tolerances
        if best_x is not None and calls < budget - 3:
            if time.time() < deadline:
                try:
                    result = minimize(
                        counting_func,
                        best_x,
                        method='Powell',
                        bounds=bounds,
                        options={'maxiter': 300, 'xtol': 1e-12, 'ftol': 1e-14}
                    )
                    if result.fun < best_f:
                        update_best(result.x, result.fun)
                except Exception:
                    pass

        # Ensure we don't exceed budget
        if best_x is None:
            # Fallback: evaluate at origin
            best_x = np.zeros(dim)
            best_f = counting_func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
