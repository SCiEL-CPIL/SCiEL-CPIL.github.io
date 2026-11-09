"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    (1+1)-ES with 1/5 success rule approximation.
    
    This approach is based on the high-performing Inspiration 1, which
    outperformed the hybrid strategy of the current program. It uses a
    single trajectory of evolution strategy with adaptive step size.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Time guard for safety, though computation is very fast
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit
        
        rng = np.random.default_rng(0)
        
        # Initialize with a random point
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        
        sigma = 2.0
        calls_made = 1
        
        # Run (1+1)-ES for the remaining budget
        while calls_made < self.budget:
            # Check time limit
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls_made += 1
            
            # Update if better
            if fy < fx:
                x, fx = y, fy
                # Increase step size on success (aggressive, as in Inspiration 1)
                sigma *= 1.5
            else:
                # Decrease step size on failure
                sigma *= 0.87
                
        return x, fx
# EVOLVE-BLOCK-END
