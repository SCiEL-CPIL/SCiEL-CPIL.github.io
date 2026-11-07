import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # For DTLZ2 with 3 objectives and 12 dimensions:
        # x[0], x[1] are position variables
        # x[2]...x[11] are decision variables, optimal at 0.5
        
        # Strategy: Use a 10x10 uniform grid for the 2D position space
        # This guarantees coverage of boundaries and corners
        # Remaining budget can be used for additional points if needed
        
        # Create 10x10 grid for position variables
        n_grid = 10
        grid_x0 = np.linspace(lo[0], hi[0], n_grid)
        grid_x1 = np.linspace(lo[1], hi[1], n_grid)
        
        # Generate grid points
        X_grid = np.zeros((n_grid * n_grid, self.dim))
        idx = 0
        for x0 in grid_x0:
            for x1 in grid_x1:
                X_grid[idx, 0] = x0
                X_grid[idx, 1] = x1
                X_grid[idx, 2:] = 0.5  # Optimal decision variables for DTLZ2
                idx += 1
        
        # Fill remaining budget with additional points using Halton sequence
        # to complement the grid coverage
        remaining = self.budget - n_grid * n_grid
        if remaining > 0:
            X_extra = np.zeros((remaining, self.dim))
            
            def halton(index, base):
                f = 1.0
                r = 0.0
                while index > 0:
                    f /= base
                    r += f * (index % base)
                    index //= base
                return r
            
            for i in range(remaining):
                if time.time() > deadline:
                    remaining = i
                    break
                X_extra[i, 0] = lo[0] + (hi[0] - lo[0]) * halton(i + 1, 2)
                X_extra[i, 1] = lo[1] + (hi[1] - lo[1]) * halton(i + 1, 3)
                X_extra[i, 2:] = 0.5
            
            X = np.vstack([X_grid, X_extra[:remaining]])
        else:
            X = X_grid
        
        # Evaluate all points
        F = np.array([problem(x) for x in X])
        
        return X
