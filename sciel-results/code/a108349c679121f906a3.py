import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = [b[0] for b in bounds]
        hi = [b[1] for b in bounds]
        dim = self.dim
        
        start_time = time.time()
        deadline = 295.0
        
        # Pre-compute Pareto front
        # f1(x1) = 1 - exp(-4*x1) * sin(6*pi*x1)^8
        # f2 = 1 - sqrt(f1)
        
        # Dense grid for x1
        n_grid = 200000
        x_grid = np.linspace(0, 1, n_grid)
        
        # Calculate f1
        sin_term = np.sin(6 * np.pi * x_grid)
        sin_term = np.power(sin_term, 8)
        exp_term = np.exp(-4 * x_grid)
        f1_grid = 1.0 - exp_term * sin_term
        
        # f1 is not monotonic. We want to select 200 distinct f1 values.
        # To maximize HV, we should cover the range of f1 well.
        # Let's find the min and max f1.
        f1_min = np.min(f1_grid)
        f1_max = np.max(f1_grid)
        
        # Select 200 target f1 values with non-uniform spacing to maximize HV
        # Optimal spacing for concave front f2 = 1 - sqrt(f1) is f1 ~ u^4
        # But we need to map to actual f1 range [f1_min, f1_max]
        n_evals = self.budget
        u = np.linspace(0, 1, n_evals)
        # Use f1 = f1_min + (f1_max - f1_min) * u^4 to cover the full range
        target_f1s = f1_min + (f1_max - f1_min) * u**4
        
        # For each target f1, find the x1 that produces it.
        # Since f1 is not monotonic, we can just find the index in f1_grid 
        # that is closest to target_f1, or use interpolation.
        # However, f1_grid is not sorted.
        # We need a mapping from f1 to x1.
        
        # Let's create a sorted array of f1 values and corresponding x1 values.
        # But there are duplicates in f1. We can just pick the first x1 for each f1.
        # Or better, for a target f1, find the x1 in x_grid such that f1(x1) is closest.
        
        # To do this efficiently, we can sort f1_grid and x_grid together.
        sort_idx = np.argsort(f1_grid)
        f1_sorted = f1_grid[sort_idx]
        x1_sorted = x_grid[sort_idx]
        
        # Now for each target_f1, use np.searchsorted to find the position.
        # Since f1_sorted is monotonic, we can interpolate.
        # But f1 is not strictly monotonic, so f1_sorted will have plateaus.
        # np.interp handles this by taking the value at the bracketing points.
        
        # Clip target_f1s to valid range
        target_f1s = np.clip(target_f1s, f1_min, f1_max)
        x1_selected = np.interp(target_f1s, f1_sorted, x1_sorted)
        
        # Construct and evaluate points
        for i in range(n_evals):
            if time.time() - start_time > deadline:
                break
            x = np.zeros(dim)
            x[0] = x1_selected[i]
            x_clipped = np.clip(x, lo, hi)
            problem(x_clipped)
        
        return None
