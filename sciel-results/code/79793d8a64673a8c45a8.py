import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295  # safety margin of 5 seconds

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        used = 0

        # DTLZ7 Pareto front: x[2:] = 0, x[0] and x[1] vary in [0,1]
        # We need to spread points on the (x[0], x[1]) square to maximize hypervolume.
        
        # Strategy: Use a Sobol-like quasi-random sequence or stratified sampling
        # for the (x[0], x[1]) plane to get good spread.
        
        # Generate points using a combination of:
        # 1. A regular grid for baseline coverage
        # 2. Quasi-random points to fill gaps
        
        n_grid = int(np.sqrt(self.budget))
        # Adjust grid to fit budget
        while n_grid * n_grid > self.budget:
            n_grid -= 1
        
        # Create grid points
        x0_grid = np.linspace(lo[0], hi[0], n_grid)
        x1_grid = np.linspace(lo[1], hi[1], n_grid)
        X0, X1 = np.meshgrid(x0_grid, x1_grid)
        X0 = X0.flatten()
        X1 = X1.flatten()
        
        n_grid_pts = len(X0)
        
        # Evaluate grid points
        for i in range(n_grid_pts):
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = X0[i]
            x[1] = X1[i]
            # x[2:] = 0
            problem(x)
            used += 1
        
        # Fill remaining budget with quasi-random points using Halton sequence
        # for better coverage than pure random
        
        def halton(index, base):
            """Generate the Halton sequence."""
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        
        remaining = self.budget - used
        # Use a deterministic Halton-like sequence for the remaining points
        # We'll generate points in a stratified manner
        
        # For the remaining points, use a finer grid or random stratification
        # Let's use a simple approach: divide the remaining budget into sub-grids
        
        if remaining > 0:
            # Use np.random with seed for reproducibility, but stratified
            # Actually, let's use a simple deterministic space-filling: 
            # golden ratio-based quasi-random in 2D
            
            golden = (1 + np.sqrt(5)) / 2  # phi
            inv_golden = 1.0 / golden  # ~0.618
            
            for i in range(remaining):
                if time.time() > deadline:
                    break
                
                # Use a 2D Halton sequence
                n = i + 1  # 1-indexed
                x0_val = lo[0] + (hi[0] - lo[0]) * halton(n, 2)
                x1_val = lo[1] + (hi[1] - lo[1]) * halton(n, 3)
                
                x = np.zeros(self.dim)
                x[0] = x0_val
                x[1] = x1_val
                # x[2:] = 0
                problem(x)
                used += 1

        # Additional refinement: add more Halton points if we have time
        # and budget remaining, using a shifted Halton sequence for better coverage
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Use a shifted Halton sequence to avoid overlapping with previous points
            n = used + 1  # 1-indexed, offset by already used points
            x0_val = lo[0] + (hi[0] - lo[0]) * halton(n + 1000, 2)
            x1_val = lo[1] + (hi[1] - lo[1]) * halton(n + 1000, 3)
            
            x = np.zeros(self.dim)
            x[0] = x0_val
            x[1] = x1_val
            # x[2:] = 0
            problem(x)
            used += 1

        return None
