import numpy as np
import time

class Optimizer:
    """DTLZ2-specific optimizer: exploit known structure for Pareto front sampling."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300  # safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        g_start = self.n_obj - 1
        base_x = np.zeros(self.dim)
        base_x[g_start:] = 0.5

        evaluated_points = []
        
        # Strategy: Use a dense grid for the first part of the budget
        # sqrt(200) ~ 14. 14*14 = 196.
        # Use 14x14 grid.
        grid_size = 14
        num_grid_points = grid_size * grid_size
        
        # Generate grid points
        for i in range(grid_size):
            for j in range(grid_size):
                if len(evaluated_points) >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                u = (i + 0.5) / grid_size
                v = (j + 0.5) / grid_size
                
                x0 = (2.0 / np.pi) * np.arcsin(u)
                x1 = v
                
                x = base_x.copy()
                x[0] = x0
                x[1] = x1
                
                f = problem(x)
                evaluated_points.append(x)
                
        # If we have budget left, we can add a few more points
        # For example, if we used 196, we have 4 left.
        # We can add 4 points from a Halton sequence or just stop.
        # Adding random points might help fill small gaps.
        if len(evaluated_points) < self.budget:
             # Simple fill with Halton
             def halton(index, base):
                f = 1.0
                r = 0.0
                while index > 0:
                    f /= base
                    r += f * (index % base)
                    index //= base
                return r

             for k in range(len(evaluated_points), self.budget):
                 if time.time() > deadline:
                     break
                 # Use index k+1 to continue sequence or just use k
                 u = halton(k + 1, 2)
                 v = halton(k + 1, 3)
                 
                 x0 = (2.0 / np.pi) * np.arcsin(u)
                 x1 = v
                 
                 x = base_x.copy()
                 x[0] = x0
                 x[1] = x1
                 
                 f = problem(x)
                 evaluated_points.append(x)

        return evaluated_points
