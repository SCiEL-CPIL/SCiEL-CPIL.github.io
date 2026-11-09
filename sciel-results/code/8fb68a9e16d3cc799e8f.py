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
        
        # Initialize with a few random samples to get a good starting point
        best_x = rng.uniform(-4.0, 4.0, dim)
        best_f = func(best_x)
        calls = 1
        
        # Also try origin and a few other points
        test_points = [np.zeros(dim), rng.uniform(-4.0, 4.0, dim), rng.uniform(-4.0, 4.0, dim)]
        for tp in test_points:
            if calls >= budget:
                break
            if time.time() > deadline:
                break
            ft = func(tp)
            calls += 1
            if ft < best_f:
                best_f = ft
                best_x = tp.copy()
        
        # Current state for the local search
        x_curr = best_x.copy()
        f_curr = best_f
        sigma = 1.0
        
        # 1/5 success rule parameters
        window_size = 15
        success_count = 0
        success_window = deque(maxlen=window_size)
        
        # Restart parameters
        no_improve_count = 0
        max_no_improve = 30
        restart_sigma = 1.0
        
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
                    # Decreasing exploration probability, increasing perturbation
                    # Early on: more likely to restart from random
                    # Later: more likely to restart from best with larger perturbation
                    progress_ratio = calls / budget
                    random_restart_prob = max(0.2, 0.8 - 0.6 * progress_ratio)
                    
                    if rng.random() < random_restart_prob:
                        # Random restart
                        x_curr = rng.uniform(-4.0, 4.0, dim)
                        f_curr = func(x_curr)
                        calls += 1
                        if f_curr < best_f:
                            best_f = f_curr
                            best_x = x_curr.copy()
                    else:
                        # Restart from best with perturbation
                        x_curr = best_x + restart_sigma * rng.standard_normal(dim)
                        x_curr = np.clip(x_curr, -5.0, 5.0)
                        f_curr = func(x_curr)
                        calls += 1
                        if f_curr < best_f:
                            best_f = f_curr
                            best_x = x_curr.copy()
                    
                    # Increase restart sigma for next restart
                    restart_sigma *= 1.5
                    restart_sigma = min(restart_sigma, 5.0)
                    
                    # Reset counters
                    no_improve_count = 0
                    max_no_improve = int(30 + 20 * progress_ratio)  # Allow more exploration later
                    sigma = 1.0
                    success_window.clear()
                    
                    if calls >= budget:
                        break

        return best_x, best_f
