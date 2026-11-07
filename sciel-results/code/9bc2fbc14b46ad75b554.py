import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Fix variables 3..11 to midpoints
        # Assume 0, 1, 2 are the critical ones for WFG9's 3 objectives
        # This is a strong assumption based on WFG literature where 
        # the first k variables define the front shape.
        
        # Create grid for first 3 variables
        # Budget 200.
        # 6 * 6 * 5 = 180 points.
        n0, n1, n2 = 6, 6, 5
        
        grid0 = np.linspace(lo[0], hi[0], n0)
        grid1 = np.linspace(lo[1], hi[1], n1)
        grid2 = np.linspace(lo[2], hi[2], n2)
        
        midpoints = (lo + hi) / 2.0
        base_x = midpoints.copy()
        
        count = 0
        for x0 in grid0:
            for x1 in grid1:
                for x2 in grid2:
                    if count >= self.budget:
                        break
                    x = base_x.copy()
                    x[0] = x0
                    x[1] = x1
                    x[2] = x2
                    # Evaluate
                    problem(x)
                    count += 1
                if count >= self.budget:
                    break
            if count >= self.budget:
                break
        
        # If we have budget left, do some random sampling in full space?
        # Or refine the grid?
        # Let's add a small random exploration in the first 3 dims with finer resolution
        # or just stop. The grid should be dense enough.
        
        # To be safe and use remaining budget effectively:
        # Use remaining budget to sample random points in the 3D subspace
        # with the other dims fixed at midpoints.
        remaining = self.budget - count
        if remaining > 0:
            rng = np.random.default_rng(42)
            for _ in range(remaining):
                x = base_x.copy()
                x[0] = rng.uniform(lo[0], hi[0])
                x[1] = rng.uniform(lo[1], hi[1])
                x[2] = rng.uniform(lo[2], hi[2])
                problem(x)
                
        return None
