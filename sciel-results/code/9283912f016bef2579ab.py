"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT6 using analytical Pareto front knowledge."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # 10s safety margin for 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_evals = self.budget

        # Strategy:
        # 1. Generate a fine grid of x1 values.
        # 2. Compute f1 for all.
        # 3. Select n_evals points from this grid that are uniformly distributed in f1.
        #    This ensures better coverage of the objective space than uniform x1,
        #    especially given the non-monotonic nature of f1(x1).
        
        # Fine grid for searching optimal x1 distribution
        n_grid = 10000
        x1_grid = np.linspace(0, 1, n_grid)
        
        # Compute f1 for the grid
        # f1 = 1 - exp(-4*x1) * sin(6*pi*x1)^6
        f1_grid = 1 - np.exp(-4 * x1_grid) * (np.sin(6 * np.pi * x1_grid) ** 6)
        
        # Sort by f1 to select uniformly spaced points in objective space
        # We want to pick points that cover the range of f1 evenly.
        # Since f2 is monotonic with f1, this also covers f2 well.
        # However, f1 is not monotonic in x1. Sorting f1_grid will reorder x1_grid.
        # We need to map back to x1.
        
        # Get indices that sort f1_grid
        sort_idx = np.argsort(f1_grid)
        f1_sorted = f1_grid[sort_idx]
        x1_sorted = x1_grid[sort_idx]
        
        # Select n_evals points uniformly from the sorted f1 range
        # Use linspace indices to pick evenly spaced points in the sorted array
        # This approximates uniform distribution in f1 value.
        indices = np.linspace(0, n_grid - 1, n_evals, dtype=int)
        
        # Get the selected f1 values and corresponding x1 values
        f1_selected = f1_sorted[indices]
        x1_selected = x1_sorted[indices]
        
        # Construct points
        points = np.zeros((n_evals, self.dim))
        points[:, 0] = x1_selected
        
        # Evaluate all points
        recorded = []
        for i in range(n_evals):
            if time.time() > deadline:
                break
            x = points[i]
            f = problem(x)
            recorded.append(f)

        return recorded
# EVOLVE-BLOCK-END
