import numpy as np


class Optimizer:
        def __init__(self, budget, dim):
            self.budget = budget
            self.dim = dim

        def __call__(self, func):
            rng = np.random.default_rng(0)
            dim = self.dim
            budget = self.budget
            
            # Phase 1: Initial Sampling
            # Use 10 points to find a good basin
            x = np.zeros(dim)
            fx = func(x)
            best_x = x.copy()
            best_f = fx
            calls = 1
            
            # 9 random points
            for _ in range(9):
                x_r = rng.uniform(-5, 5, dim)
                fx_r = func(x_r)
                calls += 1
                if fx_r < best_f:
                    best_f = fx_r
                    best_x = x_r.copy()
            
            # Phase 2: (1+1)-ES
            x = best_x
            fx = best_f
            sigma = 1.0
            stagnation = 0
            
            # We have budget - calls left
            while calls < budget:
                # Check time? The task says 300s limit. 500 calls is very fast.
                
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.3
                    stagnation = 0
                else:
                    sigma *= 0.75
                    stagnation += 1
                    
                    if stagnation > 50:
                        # Restart sigma
                        sigma = max(sigma, 0.5) 
                        stagnation = 0
                        
                # Optional: If we are very close, reduce sigma faster?
                # The "two-stage polish" suggests explicit control.
                if calls > budget * 0.7:
                    # Force smaller sigma in the last 30%
                    sigma = min(sigma, 0.1)
                    if calls > budget * 0.9:
                        sigma = min(sigma, 0.01)

            return x, fx
