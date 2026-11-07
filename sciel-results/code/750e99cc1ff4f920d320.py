"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer: grid-based decomposition with local simplex search."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Phase 1: Grid-based decomposition
        # Partition the 4D space into hypercubes and evaluate centers
        # Use a 3x3x3x3 grid = 81 points for initial coverage
        grid_per_dim = 3
        n_grid = grid_per_dim ** dim
        # If grid is too large for budget, reduce
        while n_grid > budget * 0.4 and grid_per_dim > 1:
            grid_per_dim -= 1
            n_grid = grid_per_dim ** dim

        # Generate grid points (centers of hypercubes)
        grid_points = []
        for i in range(grid_per_dim):
            for j in range(grid_per_dim):
                for k in range(grid_per_dim):
                    for l in range(grid_per_dim):
                        x = np.array([
                            lo[0] + (i + 0.5) / grid_per_dim * (hi[0] - lo[0]),
                            lo[1] + (j + 0.5) / grid_per_dim * (hi[1] - lo[1]),
                            lo[2] + (k + 0.5) / grid_per_dim * (hi[2] - lo[2]),
                            lo[3] + (l + 0.5) / grid_per_dim * (hi[3] - lo[3]),
                        ])
                        grid_points.append(x)
        grid_points = np.array(grid_points)

        # Evaluate grid points
        F_grid = np.array([problem(x) for x in grid_points])
        used = n_grid

        # Add some random points for diversity
        n_random = min(20, budget - used - 10)
        if n_random > 0:
            X_rand = lo + (hi - lo) * rng.random((n_random, dim))
            F_rand = np.array([problem(x) for x in X_rand])
            used += n_random
            grid_points = np.vstack([grid_points, X_rand])
            F_grid = np.vstack([F_grid, F_rand])

        def pareto_mask(F_arr):
            """Return boolean mask of non-dominated points."""
            n = len(F_arr)
            mask = np.ones(n, dtype=bool)
            for i in range(n):
                for j in range(n):
                    if i != j and np.all(F_arr[j] <= F_arr[i]) and np.any(F_arr[j] < F_arr[i]):
                        mask[i] = False
                        break
            return mask

        def is_dominated(f, F_arr):
            """Check if f is dominated by any point in F_arr."""
            if np.any(np.all(F_arr <= f, axis=1) & np.any(F_arr < f, axis=1)):
                return True
            return False

        # Phase 2: Local simplex search on each grid cell
        # For each non-dominated grid point, run Nelder-Mead on a weighted sum
        # with different weight vectors to explore the Pareto front
        pm = pareto_mask(F_grid)
        nd_indices = np.where(pm)[0]

        # Weight vectors for multi-objective optimization (uniform on simplex)
        # For 2 objectives, weights are (w, 1-w) for w in [0, 1]
        n_weights = 10
        weights = np.linspace(0.05, 0.95, n_weights)

        # For each non-dominated point, try a few weight vectors
        max_polish_per_point = 3
        polish_budget = budget - used - 5

        if polish_budget > 0:
            # Sort non-dominated points by sum of objectives
            sums = np.sum(F_grid[pm], axis=1)
            top_nd = nd_indices[np.argsort(sums)][:min(15, len(nd_indices))]

            for idx in top_nd:
                if used >= budget - 5 or time.time() > deadline - 20:
                    break

                x0 = grid_points[idx].copy()
                f0 = F_grid[idx].copy()

                # Try a few weight vectors for this point
                n_try = min(max_polish_per_point, n_weights, polish_budget)
                # Select weight vectors based on the current objective values
                # Points with lower f1 should try weights favoring f2, and vice versa
                if n_obj == 2:
                    # Normalize objectives for weight selection
                    f_range = f0 - np.min(F_grid[pm], axis=0)
                    f_range = f_range / (np.max(F_grid[pm], axis=0) - np.min(F_grid[pm], axis=0) + 1e-10)
                    # Select weights that complement the current point
                    w_start = int(f_range[0] * n_weights)
                    selected_weights = []
                    for offset in [-1, 0, 1]:
                        w_idx = (w_start + offset) % n_weights
                        if w_idx not in selected_weights:
                            selected_weights.append(weights[w_idx])
                    selected_weights = selected_weights[:n_try]
                else:
                    selected_weights = weights[:n_try]

                for w in selected_weights:
                    if used >= budget - 5 or time.time() > deadline - 20:
                        break

                    # Define scalarized objective
                    def obj_func(x, w=w):
                        f = problem(x)
                        return w * f[0] + (1 - w) * f[1]

                    # Run Nelder-Mead local search
                    try:
                        result = minimize(obj_func, x0, method='Nelder-Mead',
                                         options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-6})
                        x_new = result.x
                        f_new = problem(x_new)
                        used += 1

                        # Add to archive if non-dominated
                        if not is_dominated(f_new, F_grid):
                            grid_points = np.vstack([grid_points, x_new])
                            F_grid = np.vstack([F_grid, f_new])
                    except:
                        pass

        # Final cleanup: remove dominated points
        if len(F_grid) > 1:
            pm_final = pareto_mask(F_grid)
            grid_points = grid_points[pm_final]
            F_grid = F_grid[pm_final]

        return grid_points
# EVOLVE-BLOCK-END
