import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit. We stop at 290s.
        deadline = start_time + 290
        
        dim = self.dim
        total_budget = self.budget
        
        # Portfolio parameters
        n_restarts = 10
        budget_per_restart = total_budget // n_restarts
        # Ensure we don't exceed total budget
        if budget_per_restart < 10:
            n_restarts = max(1, total_budget // 10)
            budget_per_restart = total_budget // n_restarts
            
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        
        calls_made = 0
        
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if calls_made >= total_budget:
                break
                
            # Random start
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
            sigma = 2.0
            
            for step in range(budget_per_restart - 1):
                if time.time() > deadline:
                    break
                if calls_made >= total_budget:
                    break
                    
                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.5
                else:
                    sigma *= 0.87
                    
        return best_x, best_f
