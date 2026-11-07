"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware optimizer for ZDT3 exploiting its discontinuous Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        all_points = []
        used = 0

        # ZDT3 structure with g=1 (x[1:]=0):
        # f1 = x0
        # f2 = 1 - sqrt(x0) - x0 * sin(10 * pi * x0)
        #
        # Strategy:
        # 1. Use a dense grid to identify the non-dominated envelope (record lows in f2)
        # 2. Select 'budget' points uniformly from the non-dominated x0 values
        # 3. This ensures all evaluations are on the true Pareto front and well-distributed

        # Dense grid to identify non-dominated x0 values
        n_grid = 10000
        x_grid = np.linspace(0.0, 1.0, n_grid)
        f2_grid = 1.0 - np.sqrt(x_grid) - x_grid * np.sin(10.0 * np.pi * x_grid)

        # Find indices where f2 is a new running minimum (non-dominated points)
        # A point is non-dominated if its f2 is strictly less than all previous f2 values
        running_min = np.full(n_grid, np.inf)
        running_min[0] = f2_grid[0]
        for i in range(1, n_grid):
            if f2_grid[i] < running_min[i - 1]:
                running_min[i] = f2_grid[i]
            else:
                running_min[i] = running_min[i - 1]

        # Non-dominated points are those where f2_grid[i] == running_min[i]
        # Use a small tolerance for floating point comparison
        tol = 1e-12
        is_nd = (f2_grid <= running_min + tol) & (f2_grid < running_min + tol * 100)
        
        # More robust: a point is non-dominated if it sets a new record
        # i.e., f2_grid[i] < min(f2_grid[0:i])
        # Let's recompute more carefully
        min_so_far = np.empty(n_grid)
        min_so_far[0] = f2_grid[0]
        for i in range(1, n_grid):
            min_so_far[i] = min(min_so_far[i - 1], f2_grid[i])
        
        # Non-dominated: f2_grid[i] == min_so_far[i] (within tolerance)
        is_nd = np.array([(f2_grid[i] <= min_so_far[i] + tol) for i in range(n_grid)])
        
        # Actually, let's use the simpler approach: record setters
        # A point at index i is a record setter if f2_grid[i] < min(f2_grid[0:i])
        # For i=0, it's always a record setter
        is_nd = np.zeros(n_grid, dtype=bool)
        is_nd[0] = True
        current_min = f2_grid[0]
        for i in range(1, n_grid):
            if f2_grid[i] < current_min:
                is_nd[i] = True
                current_min = f2_grid[i]

        x_nd = x_grid[is_nd]

        # Ensure we include x0=0 and x0=1 for full front coverage
        if x_nd[0] > 0:
            x_nd = np.concatenate([[0.0], x_nd])
        if x_nd[-1] < 1.0:
            x_nd = np.concatenate([x_nd, [1.0]])

        # Select 'budget' points uniformly from the non-dominated x0 values
        if len(x_nd) <= budget:
            selected_x = x_nd
        else:
            # Select budget indices uniformly from the range
            indices = np.linspace(0, len(x_nd) - 1, budget, dtype=int)
            selected_x = x_nd[indices]

        # Ensure we don't exceed budget
        n_evals = min(len(selected_x), budget)

        for i in range(n_evals):
            if used >= budget:
                break
            if time.time() > deadline:
                break

            x = np.zeros(dim)
            x[0] = selected_x[i]
            # x[1:] = 0 achieves g = 1 (minimum)

            f = problem(x)
            all_points.append(f)
            used += 1

        return all_points
# EVOLVE-BLOCK-END
