import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42) # Fixed seed for reproducibility in dev, but task says hidden seed. 
        # The task says "hidden seeded shift...". The optimizer itself should be deterministic or use its own RNG.
        # Using a fixed RNG seed in the optimizer is fine and often preferred for stability.
        
        start_time = time.time()
        deadline = start_time + 320 # Safety margin from 330s
        
        # State variables
        x_best = rng.uniform(-5, 5, self.dim)
        f_best = func(x_best)
        
        # Current search state
        x_curr = x_best.copy()
        f_curr = f_best
        sigma = 1.0 # Initial step size
        
        # Restart parameters
        max_failures = 20 # If 20 steps in a row fail, restart
        failures = 0
        
        calls_made = 1
        
        for _ in range(self.budget - 1):
            # Time check
            if time.time() > deadline:
                break
            
            # Generate candidate
            step = sigma * rng.standard_normal(self.dim)
            y = np.clip(x_curr + step, -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < f_curr:
                # Success
                x_curr = y
                f_curr = fy
                sigma *= 1.2 # Increase step size
                failures = 0
                
                # Update global best
                if fy < f_best:
                    f_best = fy
                    x_best = y.copy()
            else:
                # Failure
                sigma *= 0.8 # Decrease step size
                failures += 1
                
            # Restart logic
            if failures > max_failures:
                # Reset to a new random point
                # We keep x_best to return at the end, but we reset the local search
                x_curr = rng.uniform(-5, 5, self.dim)
                f_curr = func(x_curr)
                calls_made += 1
                sigma = 1.0
                failures = 0
                
                # If the new random point is better, update best
                if f_curr < f_best:
                    f_best = f_curr
                    x_best = x_curr.copy()

        return x_best, f_best
