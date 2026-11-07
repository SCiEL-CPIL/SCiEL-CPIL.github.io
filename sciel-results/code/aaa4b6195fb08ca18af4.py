"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule adapted for BBOB f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill
        
        # Use seed 0 as in the highest scoring inspiration
        rng = np.random.default_rng(0)
        
        # Initialize in the full box [-5, 5]
        x = rng.uniform(-5.0, 5.0, self.dim)
        fx = float(func(x))
        
        # Initial step size
        sigma = 2.0
        
        # Track best found (though (1+1) keeps current best if it's the only one)
        best_x = x.copy()
        best_fx = fx
        
        # Perform budget - 1 additional evaluations
        for i in range(self.budget - 1):
            # Check time limit
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5.0, 5.0)
            fy = float(func(y))
            
            # Update logic
            if fy < fx:
                x = y
                fx = fy
                # Increase step size on success
                sigma *= 1.5
                # Update global best
                if fx < best_fx:
                    best_x = x.copy()
                    best_fx = fx
            else:
                # Decrease step size on failure
                sigma *= 0.87
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
