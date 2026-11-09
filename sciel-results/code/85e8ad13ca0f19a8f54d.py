import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        # Safety margin: 280s out of 300s limit
        deadline = start_time + 280.0

        rng = np.random.default_rng(42)

        # Start at the center of the box, which is a strong prior for the minimum location
        # The minimizer is in [-4, 4]^5, so 0 is a very good guess.
        x = np.zeros(dim)
        fx = func(x)
        
        # Track best found
        best_x = x.copy()
        best_f = fx

        # Initial step size
        # The domain is [-5, 5], so a step of 2.0 covers a significant portion
        sigma = 2.0
        
        # Adaptation factors for 1/5 success rule
        # Standard values: increase by 1.1, decrease by 1/1.1
        c_inc = 1.1
        c_dec = 1.0 / 1.1
        
        calls_made = 1

        for i in range(budget - 1):
            # Check deadline
            if time.time() > deadline:
                break
            
            # Generate candidate
            # Spherical step: y = x + sigma * z
            step = sigma * rng.standard_normal(dim)
            y = np.clip(x + step, -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < fx:
                # Success: move to new point, increase step size
                x = y
                fx = fy
                sigma *= c_inc
                
                # Update best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                # Failure: stay, decrease step size
                sigma *= c_dec

        return best_x, best_f
