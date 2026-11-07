"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-Start (1+1)-ES with adaptive sigma and stagnation-based restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 280.0  # Safety margin for 300s limit
        
        rng = np.random.default_rng(0)
        
        # Initialize global best
        x_best = rng.uniform(-5, 5, self.dim)
        f_best = func(x_best)
        calls = 1
        
        # Parameters for (1+1)-ES
        sigma = 2.0
        stagnation_limit = 30  # Restart if no improvement for this many steps
        stagnation_count = 0
        
        # Current local search state
        x_cur = x_best.copy()
        f_cur = f_best
        
        while calls < self.budget:
            # Time check
            if time.time() - start_time > time_limit:
                break
                
            # Generate mutant
            y = x_cur + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < f_cur:
                # Improvement in local search
                x_cur, f_cur = y, fy
                sigma *= 1.5
                stagnation_count = 0
                
                if f_cur < f_best:
                    x_best, f_best = y, fy
            else:
                # No improvement
                sigma *= 0.8
                stagnation_count += 1
                
                if stagnation_count >= stagnation_limit:
                    # Restart from a new random point
                    # Occasionally restart near the best to exploit, otherwise explore
                    if rng.random() < 0.2 and calls < self.budget * 0.8:
                        # Exploit around best
                        x_cur = x_best + 1.0 * rng.standard_normal(self.dim)
                        x_cur = np.clip(x_cur, -5, 5)
                        f_cur = func(x_cur)
                        calls += 1
                    else:
                        # Explore new region
                        x_cur = rng.uniform(-5, 5, self.dim)
                        f_cur = func(x_cur)
                        calls += 1
                    
                    sigma = 2.0
                    stagnation_count = 0
                    
                    if f_cur < f_best:
                        x_best, f_best = x_cur, f_cur

        return x_best, f_best
# EVOLVE-BLOCK-END
