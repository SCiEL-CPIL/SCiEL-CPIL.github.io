"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with refined step size adaptation and wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        # 330s wall-clock limit per evaluation; use 320s as safety margin
        deadline = time.time() + 320.0
        rng = np.random.default_rng(0)
        
        # Initialize
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        
        # Initial sigma: 2.0 is reasonable for [-5,5] domain
        sigma = 2.0
        
        # Adaptation factors
        # Slightly more conservative increase to prevent overshooting in flat areas
        # Faster decrease to lock in when close
        inc_factor = 1.2
        dec_factor = 0.9
        min_sigma = 1e-5
        max_sigma = 10.0
        
        for _ in range(self.budget - 1):
            if time.time() >= deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            
            if fy < fx:
                x, fx = y, fy
                sigma = min(sigma * inc_factor, max_sigma)
            else:
                sigma = max(sigma * dec_factor, min_sigma)
                
        return x, fx
# EVOLVE-BLOCK-END
