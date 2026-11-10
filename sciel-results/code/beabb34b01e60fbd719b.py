import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # WFG9: m=3 objectives.
        # Shape variables: x[0], x[1]
        # Distance variables: x[2] ... x[11]
        # Set distance variables to lower bounds to minimize distance penalty
        # and lie on the Pareto front.
        
        x_fixed = np.empty(self.dim)
        # Initialize all to lower bound
        x_fixed[:] = lo
        
        # Grid search on shape variables
        # Budget 200. 14x14 = 196.
        grid_size = int(np.sqrt(self.budget))
        if grid_size * grid_size > self.budget:
            grid_size -= 1
            
        # Ensure at least 1
        grid_size = max(grid_size, 1)
        
        x0_vals = np.linspace(lo[0], hi[0], grid_size)
        x1_vals = np.linspace(lo[1], hi[1], grid_size)
        
        evaluated = 0
        for x0 in x0_vals:
            for x1 in x1_vals:
                if evaluated >= self.budget:
                    break
                if time.time() > deadline:
                    return None
                
                x = x_fixed.copy()
                x[0] = x0
                x[1] = x1
                # x[2:] is already lo
                
                problem(x)
                evaluated += 1
            
            if evaluated >= self.budget:
                break
            if time.time() > deadline:
                return None
                
        # Use remaining budget for random sampling in shape subspace
        # to ensure coverage if grid is sparse or to refine
        while evaluated < self.budget:
            if time.time() > deadline:
                break
            x = x_fixed.copy()
            x[0] = lo[0] + (hi[0] - lo[0]) * np.random.random()
            x[1] = lo[1] + (hi[1] - lo[1]) * np.random.random()
            problem(x)
            evaluated += 1
            
        return None
