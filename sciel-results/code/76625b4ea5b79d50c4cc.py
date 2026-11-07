import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Deadline guard: 300s limit, use 290s to be safe
        deadline = time.time() + 290.0
        
        # Initialize at center (0,0,...,0) as per directive
        x = np.zeros(self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        
        sigma = 2.0
        
        # (1+1)-ES with 1/5 success rule
        for i in range(self.budget - 1):
            if time.time() > deadline:
                break
            
            # Generate offspring
            y = x + sigma * np.random.standard_normal(self.dim)
            
            # Evaluate
            fy = func(y)
            
            # Update best
            if fy < best_fx:
                best_fx = fy
                best_x = y.copy()
                
            # Adapt step size
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.5
            else:
                sigma *= 0.87
                
            # Optional: If sigma becomes too small or too large, handle?
            # The 1/5 rule usually keeps it in a reasonable range.
            
        return best_x, best_fx
