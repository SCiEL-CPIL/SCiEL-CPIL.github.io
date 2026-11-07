"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, time guard, and adaptive sigma."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        d = self.dim
        
        # Initial point
        x = rng.uniform(-5, 5, d)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        sigma = 2.0
        
        # Adaptive sigma bounds
        sigma_min = 1e-6
        sigma_max = 10.0
        
        calls_made = 1
        
        for _ in range(self.budget - 1):
            # Time check with safety margin
            if time.time() > deadline:
                break
            
            # Generate step
            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)
            calls_made += 1
            
            # Update best
            if fy < best_fx:
                best_fx = fy
                best_x = y.copy()
                
            if fy < fx:
                # Move
                x, fx = y, fy
                # Increase sigma on success (1/5 rule variant: multiply by 1.1)
                sigma *= 1.1
                if sigma > sigma_max:
                    sigma = sigma_max
            else:
                # Decrease sigma on failure (1/5 rule variant: multiply by 0.8)
                sigma *= 0.8
                if sigma < sigma_min:
                    sigma = sigma_min
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
