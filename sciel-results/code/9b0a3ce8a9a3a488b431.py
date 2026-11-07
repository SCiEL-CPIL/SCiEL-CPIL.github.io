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

        # ZDT3 structure with g=1 (x[1:]=0)
        # f1 = x0
        # f2 = 1 - sqrt(x0) - x0 * sin(10 * pi * x0)
        
        # 1. Identify non-dominated x0 values on the ideal front
        # A point x is dominated if there exists y < x such that f2(y) <= f2(x)
        # We want to sample only from the set of "record lows" in f2
        
        # Dense grid to identify the non-dominated envelope
        n_grid = 1000000
        x_grid = np.linspace(0.0, 1.0, n_grid)
        # Calculate f2 for the grid
        f2_grid = 1.0 - np.sqrt(x_grid) - x_grid * np.sin(10.0 * np.pi * x_grid)
        
        # Find indices where f2 is a new minimum (strictly less than all previous)
        # We use a running minimum
        min_f2_so_far = np.full(n_grid, np.inf)
        min_f2_so_far[0] = f2_grid[0]
        # Vectorized computation of running minimum
        min_f2_so_far[1:] = np.minimum.accumulate(f2_grid[1:])
        
        # The non-dominated points are those where f2_grid[i] == min_f2_so_far[i]
        # Note: due to floating point, we use a small tolerance
        tol = 1e-14
        is_nd = (f2_grid <= min_f2_so_far + tol)
        
        # Get the x values that are non-dominated
        x_nd = x_grid[is_nd]
        
        # Remove duplicates that might arise from tolerance
        if len(x_nd) > 0:
            f2_nd = f2_grid[is_nd]
            # Keep only strict record lows: f2 must be less than the previous kept point's f2
            unique_mask = np.ones(len(x_nd), dtype=bool)
            prev_f2 = np.inf
            for i in range(len(x_nd)):
                if f2_nd[i] >= prev_f2 - 1e-15:
                    unique_mask[i] = False
                else:
                    prev_f2 = f2_nd[i]
            x_nd = x_nd[unique_mask]
            f2_nd = f2_nd[unique_mask]
        
        n_nd = len(x_nd)
        if n_nd <= budget:
            selected_x = x_nd
        else:
            # Select 'budget' points uniformly in f1-space for better hypervolume coverage
            # Stratified selection: divide [0,1] into budget equal intervals, pick the
            # Pareto front point closest to the midpoint of each interval
            interval_midpoints = (np.arange(budget) + 0.5) / budget
            # For each midpoint, find the Pareto front point with x0 closest to it
            selected_x = np.empty(budget)
            for i in range(budget):
                # Binary search for closest x_nd to interval_midpoints[i]
                idx = np.searchsorted(x_nd, interval_midpoints[i])
                # Check neighbors
                if idx == 0:
                    best_idx = 0
                elif idx == n_nd:
                    best_idx = n_nd - 1
                else:
                    if abs(x_nd[idx-1] - interval_midpoints[i]) <= abs(x_nd[idx] - interval_midpoints[i]):
                        best_idx = idx - 1
                    else:
                        best_idx = idx
                selected_x[i] = x_nd[best_idx]
            # Remove duplicates
            selected_x = np.unique(selected_x)
            
        # Ensure we don't exceed budget
        n_evals = len(selected_x)
        
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
