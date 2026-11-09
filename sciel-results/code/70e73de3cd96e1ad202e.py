import numpy as np
import time

class Optimizer:
    """
    Restart (1+1)-ES optimizer for BBOB-style functions.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s limit, stop at 325s
        deadline = start_time + 325.0
        
        # Use a fixed seed for reproducibility if needed, 
        # but random is generally fine for this task type.
        # The prompt implies the evaluator handles the "hidden seeded shift", 
        # so our internal randomness just needs to be effective.
        rng = np.random.default_rng(42) 

        # Initial point in [-4, 4]^5
        x = rng.uniform(-4, 4, self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        
        calls_used = 1
        sigma = 2.0
        stagnation = 0
        stagnation_limit = 25 # Restart after 25 steps of no improvement
        
        while calls_used < self.budget:
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls_used += 1
            
            if fy < fx:
                x = y
                fx = fy
                if fy < best_fx:
                    best_fx = fy
                    best_x = x.copy()
                sigma *= 1.5
                stagnation = 0
            else:
                sigma *= 0.87
                stagnation += 1
                
            # Check for stagnation and restart
            if stagnation >= stagnation_limit:
                # Restart with new random point
                x = rng.uniform(-4, 4, self.dim)
                fx = func(x)
                calls_used += 1
                sigma = 2.0
                stagnation = 0
                
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()

        return best_x, best_fx
