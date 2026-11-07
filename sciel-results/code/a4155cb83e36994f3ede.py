import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin before 300s limit
        
        # Phase 1: Global Exploration
        # Use a smaller number of points to save budget for the effective ES phase
        # The simple ES is very effective, so we just need to get into the basin of attraction
        n_explore = min(20, self.budget // 10) 
        points = rng.uniform(-5, 5, (n_explore, self.dim))
        values = np.array([func(p) for p in points])
        
        if time.time() > deadline:
            best_idx = np.argmin(values)
            return points[best_idx], values[best_idx]

        # Select top 3 points to balance exploration and exploitation
        top_k = 3
        top_indices = np.argsort(values)[:top_k]
        top_points = points[top_indices]
        top_values = values[top_indices]
        
        # Phase 2: Local Exploitation with (1+1)-ES
        # Using the adaptation rates from the 1.0 scoring inspiration
        remaining_budget = self.budget - n_explore
        budget_per_es = remaining_budget // top_k
        
        best_x = top_points[0].copy()
        best_f = top_values[0]
        
        for i in range(top_k):
            x = top_points[i].copy()
            fx = top_values[i]
            # Start with a moderate sigma. The adaptation rule will adjust it.
            sigma = 1.0
            
            # Run (1+1)-ES for this instance
            for _ in range(budget_per_es):
                if time.time() > deadline:
                    break
                
                # Generate offspring
                y = x + sigma * rng.standard_normal(self.dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    # Use the proven adaptation rates from the 1.0 scorer
                    sigma *= 1.5
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.87
                    # Prevent sigma from becoming too small to be useful
                    if sigma < 1e-8:
                        break
            
            # Update global best
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
