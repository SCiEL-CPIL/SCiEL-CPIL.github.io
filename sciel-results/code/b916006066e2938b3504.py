import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Strategy: Exploit WFG9 structure.
        # Fix variables 3..11 to midpoints.
        # Grid search the first 3 variables.
        
        # Determine grid sizes for first 3 variables
        # We want to use as much of the budget as possible for the grid.
        # 6 * 6 * 5 = 180 points.
        # Remaining 20 points for random sampling in the 3D subspace.
        
        n0, n1, n2 = 6, 6, 5
        
        # Create grids
        # Using linspace ensures we hit the boundaries, which is good for hypervolume.
        grid0 = np.linspace(lo[0], hi[0], n0)
        grid1 = np.linspace(lo[1], hi[1], n1)
        grid2 = np.linspace(lo[2], hi[2], n2)
        
        # Base point with midpoints for variables 3..11
        midpoints = (lo + hi) / 2.0
        base_x = midpoints.copy()
        
        count = 0
        # Evaluate grid points
        for x0 in grid0:
            for x1 in grid1:
                for x2 in grid2:
                    if count >= self.budget:
                        break
                    if time.time() > deadline:
                        return None # Safety break
                    
                    x = base_x.copy()
                    x[0] = x0
                    x[1] = x1
                    x[2] = x2
                    problem(x)
                    count += 1
                if count >= self.budget or time.time() > deadline:
                    break
            if count >= self.budget or time.time() > deadline:
                break
        
        # Use remaining budget for random sampling in the first 3 dimensions
        # This helps fill gaps in the grid and potentially find better points
        # near the boundaries or in dense areas of the front.
        remaining = self.budget - count
        if remaining > 0:
            # Use a fixed seed for reproducibility in the random part? 
            # The problem statement says np.random.seed(seed) is called before Optimizer creation.
            # So we can just use np.random directly.
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                x = base_x.copy()
                x[0] = np.random.uniform(lo[0], hi[0])
                x[1] = np.random.uniform(lo[1], hi[1])
                x[2] = np.random.uniform(lo[2], hi[2])
                problem(x)
                count += 1
                
        return None
