"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, restarts, and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s limit
        
        rng = np.random.default_rng(0)
        
        # Global best
        x_best = rng.uniform(-5, 5, self.dim)
        fx_best = func(x_best)
        
        calls_made = 1
        remaining = self.budget - 1
        
        # Parameters for (1+1)-ES with restarts
        sigma = 1.0
        sigma_min = 1e-3
        sigma_max = 10.0
        stall_limit = 30  # Restart after this many unsuccessful steps
        stall_count = 0
        
        while remaining > 0 and time.time() < deadline:
            y = np.clip(x_best + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_made += 1
            remaining -= 1
            
            if fy < fx_best:
                x_best, fx_best = y, fy
                sigma = min(sigma * 1.1, sigma_max)
                stall_count = 0
            else:
                sigma = max(sigma * 0.9, sigma_min)
                stall_count += 1
                
                if stall_count >= stall_limit:
                    # Restart from a new random point
                    stall_count = 0
                    sigma = 1.0
                    x_new = rng.uniform(-5, 5, self.dim)
                    fx_new = func(x_new)
                    calls_made += 1
                    remaining -= 1
                    if fx_new < fx_best:
                        x_best, fx_best = x_new, fx_new
                    else:
                        # Continue from current best but with reset sigma
                        pass
        
        return x_best, fx_best
# EVOLVE-BLOCK-END
