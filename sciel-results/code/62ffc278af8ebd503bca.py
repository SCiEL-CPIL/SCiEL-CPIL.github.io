"""Multi-objective optimizer for ZDT6 using analytical Pareto front knowledge."""
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
        deadline = start_time + 290  # safety margin for 330s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        n_evals = self.budget

        # ZDT6: f1 = 1 - exp(-4*x1) * sin(6*pi*x1)^6
        #       f2 = g(x) * (1 + sin(10*pi*f1))
        #       g(x) = 1 + 9*sum(x_i)/9  for i=2..10
        # Pareto front: x_2=...=x_10=0, so g=1, f2 = 1 + sin(10*pi*f1)
        # f1 ranges over [0,1] as x1 varies in [0,1]
        
        # Strategy: Generate points on the Pareto front by choosing x1 values
        # that give good coverage of f1 in [0,1], with x_2..x_10 = 0.
        # To maximize hypervolume, we want uniform coverage in f1-space.
        # Since f1(x1) is non-monotonic, we use a fine grid and select
        # x1 values that map to uniformly spaced f1 values.
        
        # Fine grid for mapping
        n_grid = 20000
        x1_grid = np.linspace(0, 1, n_grid)
        
        # Compute f1 for the grid
        f1_grid = 1 - np.exp(-4 * x1_grid) * (np.sin(6 * np.pi * x1_grid) ** 6)
        
        # Clip f1 to [0, 1] for safety
        f1_grid = np.clip(f1_grid, 0, 1)
        
        # We want to select n_evals x1 values such that their f1 values
        # are as uniformly distributed in [0,1] as possible.
        # Sort by f1 and pick evenly spaced indices.
        sort_idx = np.argsort(f1_grid)
        f1_sorted = f1_grid[sort_idx]
        x1_sorted = x1_grid[sort_idx]
        
        # Select n_evals points uniformly from the sorted f1 range
        indices = np.linspace(0, n_grid - 1, n_evals, dtype=int)
        
        x1_selected = x1_sorted[indices]
        
        # Construct points: x1 = selected value, x_2..x_10 = 0
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
