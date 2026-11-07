"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0
        rng = np.random.default_rng(0)
        
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        sigma = 2.0
        success_count = 0
        window_size = 15
        successes = []
        
        for i in range(self.budget - 1):
            if time.time() > deadline - 5.0:
                break
                
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            
            if fy < fx:
                x, fx = y, fy
                success_count += 1
                successes.append(1)
            else:
                successes.append(0)
                
            if len(successes) > window_size:
                successes.pop(0)
                
            if len(successes) == window_size:
                p_success = sum(successes) / window_size
                if p_success > 1/5:
                    sigma *= 1.1
                elif p_success < 1/5:
                    sigma *= 0.9
                else:
                    sigma *= 1.0
                    
            sigma = max(1e-6, min(sigma, 10.0))

        return x, fx
# EVOLVE-BLOCK-END
