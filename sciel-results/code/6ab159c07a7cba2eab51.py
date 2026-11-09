"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    (1+1)-ES with refined 1/5 success rule approximation.
    
    Improvements over the seed:
    1. Smoother step size adaptation to prevent oscillation around the minimum.
    2. Explicit time guard (though budget is the primary constraint).
    3. Slightly more conservative initial step size to ensure stability.
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
        
        # Start with a step size that covers a significant portion of the domain
        # but is not too large to cause immediate rejection
        sigma = 2.0
        calls_made = 1
        
        # Refined multipliers
        # Success: Increase sigma, but less aggressively to avoid overshooting
        # Failure: Decrease sigma, but not too fast to allow exploration
        sigma_inc = 1.2
        sigma_dec = 0.85
        
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
                sigma *= sigma_inc
            else:
                sigma *= sigma_dec
                
        return x, fx
# EVOLVE-BLOCK-END
