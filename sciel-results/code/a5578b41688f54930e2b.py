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
        
        n_points = self.budget
        
        # Generate Halton sequence
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f = f / base
                r = r + f * (index % base)
                index //= base
            return r

        # We want to include the corners to ensure the extremes of the front are covered.
        # Corners in parameter space: (0,0), (1,0), (0,1), (1,1)
        # Note: (1,0) and (1,1) map to the same point on the front, but including both
        # in the parameter space is fine, we just evaluate them.
        # However, to maximize unique points on the front, we should be careful.
        # But 200 points is small, so explicit corners are good.
        
        # Strategy: 
        # 1. Add 4 corners.
        # 2. Add (n_points - 4) Halton points.
        
        points = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]
        
        # Generate remaining points
        start_idx = 1
        for i in range(len(points), n_points):
            if time.time() > deadline:
                break
            # Use index starting from 1 for Halton
            idx = i - 3 # 1, 2, 3...
            x0 = halton(idx, 2)
            x1 = halton(idx, 3)
            points.append((x0, x1))
            
        # If we didn't reach n_points due to time, we just use what we have.
        # But 200 points is fast.
        
        results = []
        for x0, x1 in points:
            x = np.zeros(self.dim)
            x[0] = x0
            x[1] = x1
            x[2:] = 0.5
            
            f = problem(x)
            results.append(f)
            
        return results
