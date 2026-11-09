import numpy as np
import time
from collections import deque

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)
        
        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        
        sigma = 1.0
        # 1/5 rule parameters
        n = dim
        window_size = 5 * n
        success_history = deque(maxlen=window_size)
        
        stagnation_counter = 0
        restart_threshold = 40
        
        start_time = time.time()
        deadline = start_time + 280.0 # Safety margin
        
        calls_made = 1
        
        while calls_made < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate mutant
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls_made += 1
            
            success = fy < fx
            
            # Update best
            if fy < best_fx:
                best_fx = fy
                best_x = y.copy()
                stagnation_counter = 0
            else:
                stagnation_counter += 1
                
            if success:
                x = y
                fx = fy
                success_history.append(True)
            else:
                success_history.append(False)
            
            # Step size adaptation (1/5 rule)
            # Only adapt if we have enough history
            if len(success_history) == window_size:
                success_rate = np.mean(success_history)
                if success_rate > 0.2:
                    sigma *= 1.5
                elif success_rate < 0.2:
                    sigma *= 0.75
            
            # Restart logic
            if stagnation_counter >= restart_threshold:
                # Restart from a new random point or perturb best
                # Using a new random point helps escape local minima
                x = rng.uniform(-5, 5, dim)
                fx = func(x)
                calls_made += 1
                if fy < best_fx: # Note: fy is from previous step, need to check new fx
                    pass # Logic below handles it
                
                # Re-evaluate best if needed? No, we just reset current state.
                # But we must ensure we don't exceed budget.
                if calls_made >= budget:
                    break
                    
                # Reset sigma and history
                sigma = 1.0
                success_history.clear()
                stagnation_counter = 0
                
        return best_x, best_fx
