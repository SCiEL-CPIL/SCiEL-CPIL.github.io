import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Global best
        x_global = rng.uniform(-5, 5, dim)
        fx_global = func(x_global)
        calls_made = 1
        
        # Local search state
        x_local = x_global.copy()
        fx_local = fx_global
        sigma = 2.0
        min_sigma = 1e-5
        max_sigma = 10.0
        
        no_improve_local = 0
        stagnation_limit = 40 # Steps without local improvement before restart
        
        while calls_made < budget:
            if time.time() >= deadline:
                break
            
            # Generate candidate
            y = np.clip(x_local + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1
            
            if fy < fx_local:
                # Local improvement
                x_local = y
                fx_local = fy
                no_improve_local = 0
                sigma = min(sigma * 1.25, max_sigma)
                
                # Update global best
                if fx_local < fx_global:
                    x_global = x_local.copy()
                    fx_global = fx_local
            else:
                no_improve_local += 1
                sigma = max(sigma * 0.8, min_sigma)
                
                # Check for stagnation
                if no_improve_local >= stagnation_limit:
                    # Restart local search from a new random point
                    x_local = rng.uniform(-5, 5, dim)
                    fx_local = func(x_local)
                    calls_made += 1
                    no_improve_local = 0
                    sigma = 2.0
                    
                    # Update global best if this random point is better (unlikely but possible)
                    if fx_local < fx_global:
                        x_global = x_local.copy()
                        fx_global = fx_local
                        
                    if calls_made >= budget:
                        break

        return x_global, fx_global
