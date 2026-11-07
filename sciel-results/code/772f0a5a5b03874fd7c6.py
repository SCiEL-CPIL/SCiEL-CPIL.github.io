import numpy as np
import time

class Optimizer:
    """Separable ES with adaptive per-dimension step sizes and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, stop at 290s
        timeout = 290.0
        
        rng = np.random.default_rng(0)
        
        # Initialize
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        
        # Per-dimension step sizes
        # Start moderately large to explore the flat directions of Bent Cigar
        sigma = np.full(self.dim, 2.0)
        
        # Adaptation parameters
        growth = 1.25
        decay = 0.85
        sigma_min = 1e-6
        sigma_max = 15.0
        
        calls = 1
        stall_counter = 0
        stall_threshold = 50
        
        while calls < self.budget:
            if time.time() - start_time > timeout:
                break
                
            # Generate candidate
            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                # Improvement
                x = y
                fx = fy
                best_x = x.copy()
                best_fx = fy
                stall_counter = 0
                
                # Grow step sizes
                sigma *= growth
                sigma = np.clip(sigma, sigma_min, sigma_max)
            else:
                # No improvement
                # Shrink step sizes
                sigma *= decay
                sigma = np.clip(sigma, sigma_min, sigma_max)
                
                stall_counter += 1
                if stall_counter > stall_threshold:
                    # Restart from best point with randomized sigmas
                    x = best_x.copy()
                    fx = best_fx
                    # Randomize sigmas to escape potential local scaling traps
                    # Use a range that allows both exploration and exploitation
                    sigma = rng.uniform(0.5, 5.0, self.dim)
                    stall_counter = 0

        return best_x, best_fx
