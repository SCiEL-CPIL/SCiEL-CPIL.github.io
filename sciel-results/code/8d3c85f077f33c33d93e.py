import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s limit
        rng = np.random.default_rng(42)
        
        # Phase 1: Exploration - evaluate random points
        n_explore = min(50, self.budget // 10)
        explore_points = rng.uniform(-5, 5, (n_explore, self.dim))
        explore_vals = np.array([func(p) for p in explore_points])
        
        # Track best
        best_idx = np.argmin(explore_vals)
        best_x = explore_points[best_idx].copy()
        best_f = explore_vals[best_idx]
        
        # Phase 2: Exploitation from top 5 points
        n_top = min(5, n_explore)
        top_indices = np.argsort(explore_vals)[:n_top]
        top_points = explore_points[top_indices]
        top_vals = explore_vals[top_indices]
        
        remaining_budget = self.budget - n_explore
        budget_per_start = remaining_budget // n_top
        
        for i in range(n_top):
            x = top_points[i].copy()
            fx = top_vals[i]
            
            # Initialize sigma based on local scale
            sigma = 0.5
            success_count = 0
            total_trials = 0
            window = 10  # window for 1/5 rule
            
            for step in range(budget_per_start):
                if time.time() > deadline:
                    break
                
                # Generate trial
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                total_trials += 1
                
                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    sigma *= 1.2
                else:
                    sigma *= 0.85
                
                # 1/5 success rule every 10 trials
                if total_trials % window == 0:
                    if success_count > 0:
                        sigma *= (1 + 0.5 * (success_count / window - 0.2))
                    else:
                        sigma *= 0.7
                    success_count = 0
                
                # Update global best
                if fx < best_f:
                    best_x, best_f = x.copy(), fx
            
            if time.time() > deadline:
                break
        
        return best_x, best_f
