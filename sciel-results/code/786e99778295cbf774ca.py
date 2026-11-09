import numpy as np
import time
import itertools

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        d = self.dim
        budget = self.budget
        
        # Phase 1: Deterministic Grid Search Warm-up
        # The minimizer is in [-4, 4]^5.
        # We sample the grid at -4, 0, 4 to bracket the space.
        # 3^5 = 243 points.
        coords = [-4.0, 0.0, 4.0]
        grid_points = list(itertools.product(coords, repeat=d))
        
        best_x = None
        best_f = float('inf')
        count = 0
        
        for p in grid_points:
            if count >= budget:
                break
            if time.time() > deadline:
                break
            x = np.array(p)
            fx = func(x)
            count += 1
            if fx < best_f:
                best_f = fx
                best_x = x
        
        # Fallback if grid search didn't run (shouldn't happen)
        if best_x is None:
            x = np.zeros(d)
            fx = func(x)
            count = 1
        else:
            x = best_x
            fx = best_f
            
        # Phase 2: (1+1)-ES Refinement
        # Start from the best grid point
        # Initial sigma: 1.0 (half the grid spacing of 4.0)
        sigma = 1.0
        rng = np.random.default_rng(42)
        
        inc_factor = 1.1
        dec_factor = 0.9
        
        while count < budget:
            if time.time() > deadline:
                break
                
            y = x + sigma * rng.standard_normal(d)
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            count += 1
            
            if fy < fx:
                x = y
                fx = fy
                sigma *= inc_factor
            else:
                sigma *= dec_factor
                
            if sigma < 1e-10:
                break
                
        return x, fx
