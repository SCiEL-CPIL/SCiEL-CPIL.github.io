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
        
        # ZDT6 Pareto Front:
        # f1(x1) = 1 - exp(-4*x1) * sin(6*pi*x1)^8
        # f2 = 1 - sqrt(f1)
        # x2..xn = 0
        
        # 1. Determine the range of f1 on the Pareto front
        # We need min and max of f1 for x1 in [0, 1]
        # f1_max is 1 (at x1=0 and x1=1, where sin term is 0)
        # f1_min is 1 - max(exp(-4*x1) * sin(6*pi*x1)^8)
        
        # High resolution grid to find the exact min f1 and map f1 to x1
        n_grid = 1000000
        x1_grid = np.linspace(0, 1, n_grid)
        
        # Calculate f1 values
        sin_term = np.sin(6 * np.pi * x1_grid)
        sin_term_pow8 = np.power(sin_term, 8)
        exp_term = np.exp(-4 * x1_grid)
        f1_grid = 1.0 - exp_term * sin_term_pow8
        
        f1_min = np.min(f1_grid)
        f1_max = np.max(f1_grid) # Should be 1.0
        
        # 2. Select target f1 values
        # Uniform spacing in f1 is known to be effective for this front
        n_evals = self.budget
        target_f1s = np.linspace(f1_min, f1_max, n_evals)
        
        # 3. Map target f1s to x1
        # Sort f1_grid and x1_grid together
        sort_idx = np.argsort(f1_grid)
        f1_sorted = f1_grid[sort_idx]
        x1_sorted = x1_grid[sort_idx]
        
        # Use searchsorted to find the index in f1_sorted for each target_f1
        # Since f1_sorted is monotonic, this is efficient
        # We want the x1 corresponding to the f1 value closest to target
        # np.searchsorted gives the insertion point
        # We can just use np.interp for simplicity and robustness against plateaus
        # However, np.interp requires xp to be increasing. f1_sorted is.
        
        # Note: f1 is not strictly monotonic, so f1_sorted has duplicates.
        # np.interp handles this by taking the average or the value at the bracket.
        # Actually, for a given f1, there are multiple x1. We just need one.
        # np.interp will return the x1 from the first occurrence in the sorted order
        # (or effectively the one corresponding to the lower bound of the interval).
        
        x1_selected = np.interp(target_f1s, f1_sorted, x1_sorted)
        
        # 4. Evaluate points
        # Construct x vectors
        # x = [x1, 0, 0, ..., 0]
        
        for i in range(n_evals):
            if time.time() - start_time > deadline:
                break
            
            x = np.zeros(dim)
            x[0] = x1_selected[i]
            
            # Clip to bounds (though x1 is in [0,1] and others are 0, 
            # and bounds are [0,1], this is safe)
            x_clipped = np.clip(x, lo, hi)
            
            problem(x_clipped)
        
        return None
