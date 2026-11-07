"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with time-aware step control and robust best tracking."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        # Safety margin: 290s is well within the 330s kill limit.
        # This prevents any potential hang or slow-down from invalidating the run.
        deadline = time.time() + 290.0
        rng = np.random.default_rng(0)
        
        # Initialize
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        
        # Parameters from Inspiration 1 (Score 1.0)
        sigma = 2.0
        
        # Loop for remaining budget
        for _ in range(self.budget - 1):
            # Check time limit
            if time.time() > deadline:
                break
            
            # Generate candidate
            # Using standard normal noise scaled by sigma
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            
            # Update step size and position
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5  # Aggressive expansion on success
            else:
                sigma *= 0.87 # Moderate contraction on failure
            
            # Track best explicitly (redundant for strict (1+1) but safe)
            if fx < best_fx:
                best_x = x.copy()
                best_fx = fx
                
        return best_x, best_fx
# EVOLVE-BLOCK-END
