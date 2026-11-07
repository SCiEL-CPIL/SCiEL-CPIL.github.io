import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        # Phase 1: Quick initial exploration (10 points) to find a good starting region
        n_explore = 10
        explore_points = rng.uniform(-5, 5, (n_explore, self.dim))
        explore_vals = np.array([func(p) for p in explore_points])
        
        best_idx = np.argmin(explore_vals)
        x = explore_points[best_idx].copy()
        fx = explore_vals[best_idx]
        best_x = x.copy()
        best_f = fx
        
        # Phase 2: Pure (1+1)-ES with 1.5/0.87 adaptation
        # Start with moderate sigma for fast initial convergence
        sigma = 1.0
        
        # Track success rate for sigma adaptation
        window_size = 10
        successes = 0
        trials_in_window = 0
        
        remaining = self.budget - n_explore
        
        for step in range(remaining):
            if time.time() > deadline:
                break
            
            # Generate trial point
            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            
            trials_in_window += 1
            
            if fy < fx:
                x = y
                fx = fy
                successes += 1
                
                # Update global best
                if fx < best_f:
                    best_x = x.copy()
                    best_f = fx
                
                # Increase sigma on success (1.5x)
                sigma *= 1.5
            else:
                # Decrease sigma on failure (0.87x)
                sigma *= 0.87
            
            # Adapt sigma based on success rate every window_size trials
            if trials_in_window >= window_size:
                success_rate = successes / trials_in_window
                # If success rate is too low, shrink sigma more aggressively
                if success_rate < 0.1:
                    sigma *= 0.7
                elif success_rate > 0.5:
                    sigma *= 1.1
                successes = 0
                trials_in_window = 0
        
        return best_x, best_f
