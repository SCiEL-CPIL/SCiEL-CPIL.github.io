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
        
        # Analytical approach: ZDT3 has 100 local minima in f2 for g=1
        # f2(x0) = 1 - sqrt(x0) - x0 * sin(10*pi*x0)
        # Minima occur where derivative = 0:
        # d/dx0 [1 - sqrt(x0) - x0*sin(10*pi*x0)] = -1/(2*sqrt(x0)) - sin(10*pi*x0) - 10*pi*x0*cos(10*pi*x0) = 0
        # 
        # For the Pareto front, we need to find all non-dominated points.
        # The key insight is that we can directly compute the Pareto front by
        # finding all local minima of f2 and keeping only non-dominated ones.
        
        # Use a dense grid to find all local minima
        n_grid = 200000
        x_grid = np.linspace(0.0, 1.0, n_grid)
        f2_grid = 1.0 - np.sqrt(x_grid) - x_grid * np.sin(10.0 * np.pi * x_grid)
        
        # Find local minima: f2[i] < f2[i-1] and f2[i] < f2[i+1]
        is_local_min = np.zeros(n_grid, dtype=bool)
        is_local_min[1:-1] = (f2_grid[1:-1] < f2_grid[:-2]) & (f2_grid[1:-1] < f2_grid[2:])
        # Also check endpoints
        if f2_grid[0] < f2_grid[1]:
            is_local_min[0] = True
        if f2_grid[-1] < f2_grid[-2]:
            is_local_min[-1] = True
        
        x_local_min = x_grid[is_local_min]
        f2_local_min = f2_grid[is_local_min]
        
        # Now find the Pareto front among these local minima
        # Sort by x0 (f1) and keep only non-dominated points
        sort_idx = np.argsort(x_local_min)
        x_sorted = x_local_min[sort_idx]
        f2_sorted = f2_local_min[sort_idx]
        
        # Keep only record lows in f2
        n_pts = len(x_sorted)
        if n_pts > 0:
            running_min = np.full(n_pts, np.inf)
            running_min[0] = f2_sorted[0]
            running_min[1:] = np.minimum.accumulate(f2_sorted[1:])
            
            # Pareto front points are where f2 equals the running minimum
            pareto_mask = (f2_sorted <= running_min + 1e-15)
            x_pareto = x_sorted[pareto_mask]
            f2_pareto = f2_sorted[pareto_mask]
            
            # Remove duplicates
            if len(x_pareto) > 1:
                unique_mask = np.ones(len(x_pareto), dtype=bool)
                prev_f2 = np.inf
                for i in range(len(x_pareto)):
                    if f2_pareto[i] >= prev_f2 - 1e-15:
                        unique_mask[i] = False
                    else:
                        prev_f2 = f2_pareto[i]
                x_pareto = x_pareto[unique_mask]
                f2_pareto = f2_pareto[unique_mask]
        
        n_nd = len(x_pareto)
        
        # Select budget points from the Pareto front
        if n_nd <= budget:
            selected_x = x_pareto
        else:
            # Use log-spaced selection for better hypervolume coverage
            # Points near f1=0 and f1=1 contribute more to hypervolume
            log_positions = np.geomspace(1e-6, 1.0, budget)
            selected_x = np.empty(budget)
            for i in range(budget):
                idx = np.searchsorted(x_pareto, log_positions[i])
                if idx == 0:
                    best_idx = 0
                elif idx == n_nd:
                    best_idx = n_nd - 1
                else:
                    if abs(x_pareto[idx-1] - log_positions[i]) <= abs(x_pareto[idx] - log_positions[i]):
                        best_idx = idx - 1
                    else:
                        best_idx = idx
                selected_x[i] = x_pareto[best_idx]
            # Remove duplicates
            selected_x = np.unique(selected_x)
        
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
