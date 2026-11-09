import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        rng = np.random.default_rng(0)
        
        x0 = rng.uniform(-5, 5, self.dim)
        fx0 = func(x0)
        best_x = x0.copy()
        best_f = fx0
        calls_used = 1
        
        # We will run Nelder-Mead in chunks.
        # Each chunk uses a portion of the budget.
        # If progress is stagnant, we restart from a random point.
        
        while calls_used < self.budget:
            if time.time() >= deadline:
                break
            
            # Determine start point
            # 10% chance to random restart, otherwise perturb best
            if rng.random() < 0.1:
                x_start = rng.uniform(-5, 5, self.dim)
            else:
                # Perturb best with a scale that decreases over time?
                # Or just a fixed scale.
                scale = 1.0
                x_start = best_x + rng.normal(0, scale, self.dim)
                x_start = np.clip(x_start, -5, 5)
            
            # Sub-budget for this run
            # Leave some buffer for the next iteration
            remaining = self.budget - calls_used
            sub_budget = min(50, remaining) # 50 calls per NM run
            
            if sub_budget <= 0:
                break
                
            call_count = 0
            def callback(xk):
                nonlocal call_count
                call_count += 1
                if call_count >= sub_budget:
                    raise Exception("Budget")
                return False
            
            try:
                # Nelder-Mead options
                # initial_simplex_size? Not directly exposed, but we can control via xatol/fatol
                res = minimize(func, x_start, method='Nelder-Mead', 
                               options={'maxiter': sub_budget, 'xatol': 1e-6, 'fatol': 1e-6},
                               callback=callback)
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                pass
            
            calls_used += call_count
            
            # If we used all budget, break
            if calls_used >= self.budget:
                break

        return best_x, best_f
