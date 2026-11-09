import numpy as np
import time
from collections import deque

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # 15s safety margin before 300s kill
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(12345)
        
        # Phase 1: Global Exploration (20% of budget)
        # Focus on finding a point in the "inner" ellipsoid of f7
        global_budget = max(10, budget // 5)
        best_x = None
        best_f = float('inf')
        calls = 0
        
        for _ in range(global_budget):
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        if best_x is None:
            best_x = rng.uniform(-5.0, 5.0, dim)
            best_f = func(best_x)
            calls += 1

        # Phase 2: Local Refinement (80% of budget)
        # Use (1+1)-ES starting from the best global point
        x_curr = best_x.copy()
        f_curr = best_f
        sigma = 1.0
        
        # 1/5 success rule parameters
        window_size = 15
        success_count = 0
        success_window = deque(maxlen=window_size)
        
        # Restart parameters
        no_improve_count = 0
        max_no_improve = 20 # Slightly lower to allow more restarts in local phase
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate trial
            y = x_curr + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5.0, 5.0)
            fy = func(y)
            calls += 1
            
            success = fy < f_curr - 1e-12
            success_window.append(1 if success else 0)
            
            if success:
                # Improvement found
                x_curr = y
                f_curr = fy
                no_improve_count = 0
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                
                # 1/5 rule: adjust sigma based on success rate in window
                if len(success_window) >= window_size:
                    rate = sum(success_window) / window_size
                    if rate > 0.333:  # above 1/3, increase sigma
                        sigma *= 1.05
                    elif rate < 0.2:  # below 0.2, decrease sigma
                        sigma *= 0.95
                    else:
                        # in the sweet spot, keep sigma
                        pass
            else:
                no_improve_count += 1
                
                # Adaptive restart strategy
                if no_improve_count >= max_no_improve:
                    # In the local phase, restarts should be smaller perturbations
                    # or re-sampling near the best if we think we are stuck in a local minimum
                    # For f7, if we are stuck, we might be on the plateau again or in a bad spot.
                    # Let's try a moderate jump.
                    jump_size = 1.0 
                    x_curr = best_x + jump_size * rng.standard_normal(dim)
                    x_curr = np.clip(x_curr, -5.0, 5.0)
                    f_curr = func(x_curr)
                    calls += 1
                    if f_curr < best_f:
                        best_f = f_curr
                        best_x = x_curr.copy()
                    
                    # Reset counters
                    no_improve_count = 0
                    sigma = 0.5 # Reset to smaller sigma for local search
                    success_window.clear()
                    
                    if calls >= budget:
                        break

        return best_x, best_f
