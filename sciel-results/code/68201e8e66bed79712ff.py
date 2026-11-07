"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        
        # Initial point at center (0,0,0,0,0) since minimizer is in [-4,4]^5
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        
        sigma = 1.0
        
        for _ in range(self.budget - 1):
            if time.time() > deadline:
                break
                
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            
            if fy < fx:
                x, fx = y, fy
                if fy < best_fx:
                    best_x, best_fx = x.copy(), fy
                sigma *= 1.2
            else:
                sigma *= 0.9
                
            # Keep sigma within reasonable bounds
            sigma = np.clip(sigma, 0.01, 5.0)
            
        return best_x, best_fx
# EVOLVE-BLOCK-END
