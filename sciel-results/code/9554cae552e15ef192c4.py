import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        # Safety margin: The evaluator kills after 330s, limit is 300s.
        # We use 290s to be safe.
        deadline = time.time() + 290
        
        rng = np.random.default_rng(0)
        
        # Initial point: Random within the box [-5, 5]
        # Schwefel's function is multimodal, so random start is better than center.
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        
        # Initial step size: Large enough to explore the space quickly.
        # The box width is 10, so a sigma of 2.0 covers a good portion.
        sigma = 2.0
        
        calls_made = 1
        
        # (1+1)-ES with 1/5 success rule
        # Success: sigma *= 1.5
        # Failure: sigma *= 0.87
        # This ratio is chosen to maintain a success rate of approx 1/5.
        
        for _ in range(budget - 1):
            # Check time limit
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            
            # Clip to the box [-5, 5]
            y = np.clip(y, -5, 5)
            
            # Evaluate
            fy = func(y)
            calls_made += 1
            
            # Update
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.5
            else:
                sigma *= 0.87
        
        return x, fx
