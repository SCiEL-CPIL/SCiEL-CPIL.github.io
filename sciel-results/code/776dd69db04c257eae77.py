"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        # Start at the center of the box, which is a strong prior for the minimum location
        x = np.zeros(self.dim)
        fx = func(x)
        
        # Initial step size
        sigma = 2.0
        
        # Adaptation factors for 1/5 success rule
        # Using 1.1 and 1/1.1 for stability
        c_inc = 1.1
        c_dec = 1.0 / 1.1
        
        # Deadline guard: stop if we're getting close to the 300s limit
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin
        
        for _ in range(self.budget - 1):
            # Check time before each iteration
            if time.time() > deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            if fy < fx:
                x, fx = y, fy
                sigma *= c_inc
            else:
                sigma *= c_dec
        
        return x, fx
# EVOLVE-BLOCK-END
