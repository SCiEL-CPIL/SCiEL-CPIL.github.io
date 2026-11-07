import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim
        start_time = time.time()
        time_limit = 290.0 # Safety margin before 300s kill
        
        # Initialize with center
        best_x = np.zeros(d)
        best_f = func(best_x)
        calls = 1
        
        # Restarting (1+1)-ES
        # Each restart runs for a limited number of steps or until stagnation
        max_steps_per_restart = 100
        stagnation_limit = 50
        
        while calls < self.budget:
            if time.time() - start_time > time_limit:
                break
                
            # Start new restart
            # Mix of random and best-known to balance exploration/exploitation
            if rng.random() < 0.5:
                x = rng.uniform(-4, 4, d)
            else:
                x = best_x + 1.0 * rng.standard_normal(d)
                x = np.clip(x, -5, 5)
                
            fx = func(x)
            calls += 1
            if calls >= self.budget:
                if fx < best_f:
                    best_x, best_f = x, fx
                break
            
            sigma = 1.0
            no_improve = 0
            best_in_restart = fx
            best_x_restart = x.copy()
            steps_in_restart = 0
            
            while calls < self.budget and steps_in_restart < max_steps_per_restart:
                if time.time() - start_time > time_limit:
                    break
                    
                y = x + sigma * rng.standard_normal(d)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                steps_in_restart += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05 # Gentle increase
                    no_improve = 0
                    if fx < best_in_restart:
                        best_in_restart = fx
                        best_x_restart = x.copy()
                else:
                    sigma *= 0.95 # Gentle decrease
                    no_improve += 1
                    
                if no_improve > stagnation_limit:
                    break
                    
            if best_in_restart < best_f:
                best_f = best_in_restart
                best_x = best_x_restart.copy()
                
        return best_x, best_f
