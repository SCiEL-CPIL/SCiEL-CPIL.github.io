import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290
        
        best_x = None
        best_f = np.inf
        
        rng = np.random.default_rng(123)
        
        # Multi-Start Strategy
        # Split budget into 5 restarts
        num_restarts = 5
        calls_per_restart = budget // num_restarts
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Determine calls for this restart
            remaining = budget - (restart * calls_per_restart)
            alloc = remaining if restart == num_restarts - 1 else calls_per_restart
            
            if alloc < 10:
                continue
                
            # Initialize at a random point
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            
            if fx < best_f:
                best_f = fx
                best_x = x
                
            sigma = 2.5 # Larger initial step for exploration
            
            # (1+1)-ES Loop
            for _ in range(alloc - 1):
                if time.time() > deadline:
                    break
                    
                # Generate candidate
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                
                # Evaluate
                fy = func(y)
                
                # Update
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.5
                    if fx < best_f:
                        best_f = fx
                        best_x = x
                else:
                    sigma *= 0.87

        return best_x, best_f
