"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start Nelder-Mead with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # safety margin

        call_count = 0
        best_x = np.zeros(dim)
        best_fx = np.inf

        def record(x, fx):
            nonlocal best_x, best_fx, call_count
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # Phase 1: Coarse grid search to find promising regions
        # For dim=5, use a stratified grid
        n_per_dim = 3
        grid_points = np.array(np.meshgrid(*([np.linspace(-4, 4, n_per_dim)] * dim)))
        grid_points = grid_points.reshape(dim, -1).T

        # Shuffle for diversity
        rng = np.random.default_rng(42)
        perm = rng.permutation(len(grid_points))
        grid_points = grid_points[perm]

        for x0 in grid_points:
            if call_count >= budget:
                break
            if time.time() > deadline - 30:
                break
            x0_clipped = np.clip(x0, -5, 5)
            fx = func(x0_clipped)
            call_count += 1
            record(x0_clipped, fx)

        # Phase 2: Nelder-Mead from top candidates
        # Find the best few points from grid
        # We'll do NM from the global best and a few local bests

        nm_starts = []

        # Always start from global best
        nm_starts.append(best_x.copy())

        # Add a few other diverse starting points
        if call_count < budget - 50:
            # Try starting from a few random points near the best
            for _ in range(3):
                if call_count >= budget - 50:
                    break
                x_try = best_x + rng.normal(0, 1.0, dim)
                x_try = np.clip(x_try, -5, 5)
                nm_starts.append(x_try)

        # Also add a few grid points that were locally good
        # (approximated by just using a few more grid points)
        if len(grid_points) > 1 and call_count < budget - 50:
            # Take a few more grid points spread out
            step = max(1, len(grid_points) // 5)
            for idx in range(0, len(grid_points), step):
                if call_count >= budget - 50:
                    break
                x_try = grid_points[idx]
                if not np.allclose(x_try, best_x, atol=1e-6):
                    nm_starts.append(x_try.copy())
                if len(nm_starts) >= 8:
                    break

        # Run Nelder-Mead from each start
        nm_options = {
            'xatol': 1e-10,
            'fatol': 1e-12,
            'adaptive': True,
            'maxiter': 200,
        }

        for start in nm_starts:
            if call_count >= budget:
                break
            if time.time() > deadline - 5:
                break

            start_clipped = np.clip(start, -5, 5)

            def nm_func(x):
                nonlocal call_count
                x_clipped = np.clip(x, -5, 5)
                fx = func(x_clipped)
                call_count += 1
                record(x_clipped, fx)
                return fx

            try:
                result = minimize(nm_func, start_clipped, method='Nelder-Mead', options=nm_options)
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
