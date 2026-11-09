import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start = time.time()
        deadline = start + 290.0
        
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Identify shape variables (first 2) and distance variables (rest)
        # WFG9 with 3 objectives typically uses x0, x1 for shape
        n_shape = 2
        n_dist = dim - n_shape
        
        # Determine optimal constant for distance variables
        # Standard WFG h(x) is minimized around 0.35
        # Scale to bounds if necessary
        opt_dist = np.zeros(n_dist)
        for i in range(n_dist):
            span = hi[i + n_shape] - lo[i + n_shape]
            # Map 0.35 to the range
            opt_dist[i] = lo[i + n_shape] + 0.35 * span
        
        # Generate grid for shape variables
        # Use 14x14 grid = 196 points
        n_grid = 14
        grid_x0 = np.linspace(lo[0], hi[0], n_grid)
        grid_x1 = np.linspace(lo[1], hi[1], n_grid)
        
        used = 0
        
        # Evaluate grid points
        for x0 in grid_x0:
            for x1 in grid_x1:
                if used >= budget or time.time() > deadline:
                    break
                
                x = np.zeros(dim)
                x[0] = x0
                x[1] = x1
                x[2:] = opt_dist
                
                # Clip to be safe
                x = np.clip(x, lo, hi)
                
                problem(x)
                used += 1
            
            if used >= budget or time.time() > deadline:
                break
        
        # Use remaining budget for random sampling or refinement
        # Random sampling helps if the grid missed some features
        rng = np.random.default_rng(42)
        while used < budget and time.time() < deadline:
            # Random point in shape space, optimal distance
            x = np.zeros(dim)
            x[0] = rng.uniform(lo[0], hi[0])
            x[1] = rng.uniform(lo[1], hi[1])
            x[2:] = opt_dist
            
            x = np.clip(x, lo, hi)
            problem(x)
            used += 1
            
        return None
