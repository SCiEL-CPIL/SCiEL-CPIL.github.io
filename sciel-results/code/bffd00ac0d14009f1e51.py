import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        # Initialize
        best_x = rng.uniform(-5.0, 5.0, dim)
        best_f = func(best_x)
        calls = 1
        
        # Current state for the local search
        x_curr = best_x.copy()
        f_curr = best_f
        sigma = 1.0
        
        # Parameters for the Zoom In/Out strategy
        # On plateau (no improvement), we want to move far.
        # On improvement, we want to converge.
        growth_factor = 1.5
        shrink_factor = 0.5
        max_sigma = 10.0
        min_sigma = 1e-6
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate trial
            y = x_curr + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5.0, 5.0)
            fy = func(y)
            calls += 1
            
            if fy < f_curr - 1e-12:
                # Improvement found
                x_curr = y
                f_curr = fy
                sigma *= shrink_factor
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            else:
                # No improvement (likely on plateau or outside)
                sigma *= growth_factor
                
                # If sigma becomes too large, we are wandering aimlessly.
                # Restart from a new random point or the global best.
                if sigma > max_sigma:
                    # 50% chance to restart from global best, 50% from random
                    if rng.random() < 0.5:
                        x_curr = best_x.copy()
                        f_curr = best_f
                    else:
                        x_curr = rng.uniform(-5.0, 5.0, dim)
                        f_curr = func(x_curr)
                        calls += 1
                        if f_curr < best_f:
                            best_f = f_curr
                            best_x = x_curr.copy()
                    sigma = 1.0
                    if calls >= budget:
                        break

        return best_x, best_f
