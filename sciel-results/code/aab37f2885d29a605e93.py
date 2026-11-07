"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Initial point: center of box
        x = np.zeros(dim)
        fx = func(x)
        
        # Adaptive step size
        sigma = 1.0
        min_sigma = 1e-8
        max_sigma = 10.0
        
        # Track best
        best_x = x.copy()
        best_fx = fx
        
        for i in range(budget - 1):
            # Check deadline
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            
            if fy < fx:
                x, fx = y, fy
                # Increase step size on success
                sigma = min(sigma * 1.2, max_sigma)
                if fx < best_fx:
                    best_x = x.copy()
                    best_fx = fx
            else:
                # Decrease step size on failure
                sigma = max(sigma * 0.9, min_sigma)
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
