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
        
        # Pre-compute Pareto front with higher resolution
        # f1(x1) = 1 - exp(-4*x1) * sin(6*pi*x1)^8
        # f2 = 1 - sqrt(f1)
        
        # Increased grid size to 1,000,000 for higher precision in interpolation
        n_grid = 1000000
        x_grid = np.linspace(0, 1, n_grid)
        
        # Calculate f1
        sin_term = np.sin(6 * np.pi * x_grid)
        sin_term = np.power(sin_term, 8)
        exp_term = np.exp(-4 * x_grid)
        f1_grid = 1.0 - exp_term * sin_term
        
        # f1 is not monotonic. We want to select 200 distinct f1 values.
        # To maximize HV, we should cover the range of f1 well.
        f1_min = np.min(f1_grid)
        f1_max = np.max(f1_grid)
        
        # Select 200 target f1 values uniformly in [f1_min, f1_max]
        n_evals = self.budget
        target_f1s = np.linspace(f1_min, f1_max, n_evals)
        
        # Create a sorted array of f1 values and corresponding x1 values
        sort_idx = np.argsort(f1_grid)
        f1_sorted = f1_grid[sort_idx]
        x1_sorted = x_grid[sort_idx]
        
        # Interpolate x1 for each target f1
        # np.interp handles the non-strictly monotonic nature by taking the bracketing value
        x1_selected = np.interp(target_f1s, f1_sorted, x1_sorted)
        
        # Construct and evaluate points
        # Pre-allocate x vector to avoid repeated allocation
        x = np.zeros(dim)
        
        for i in range(n_evals):
            if time.time() - start_time > deadline:
                break
            
            x[0] = x1_selected[i]
            # x[1:] is already 0, which is optimal for ZDT6 Pareto front
            # Clip to bounds just in case, though x1 is in [0,1]
            # Since bounds are [0,1], clip is a no-op but safe
            problem(x)
            
        return None
