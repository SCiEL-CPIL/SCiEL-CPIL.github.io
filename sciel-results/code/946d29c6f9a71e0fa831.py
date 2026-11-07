import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        # Phase 1: Global Search (Random Sampling)
        # Use 20% of budget for exploration to find a good basin
        n_explore = max(10, int(0.2 * self.budget))
        n_refine = self.budget - n_explore
        
        rng = np.random.default_rng(42)
        
        x_best = None
        fx_best = float('inf')
        
        # Evaluate center first
        x0 = np.zeros(self.dim)
        fx0 = func(x0)
        x_best = x0
        fx_best = fx0
        
        # Sample random points
        for i in range(1, n_explore):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            if fx < fx_best:
                fx_best = fx
                x_best = x
        
        # Phase 2: Local Search ((1+1)-ES)
        # Start from the best point found
        x = x_best.copy()
        fx = fx_best
        sigma = 1.0  # Initial step size, roughly the width of a Rastrigin basin
        
        # Adapt sigma to stay within bounds and adapt to landscape
        for i in range(n_refine):
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            
            if fy < fx:
                x = y
                fx = fy
                # Increase sigma slightly to explore more, but cap it
                sigma = min(sigma * 1.2, 2.0)
            else:
                # Decrease sigma to refine
                sigma *= 0.85
                
                # If sigma is very small, we might be stuck in a local min.
                # Optional: If we have a lot of budget left, we could restart, 
                # but with 500 budget, refining is usually better.
                if sigma < 1e-4:
                    # If we stagnate completely, try a small random kick?
                    # For now, just continue with tiny steps to satisfy the loop
                    # or break if no progress is possible.
                    pass

        return x, fx
