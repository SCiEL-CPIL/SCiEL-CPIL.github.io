import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300  # Safety margin from 330s limit
        
        # Phase 1: Global Sampling
        # We use 20 random points and the center
        n_global = 20
        rng = np.random.default_rng(42) # Fixed seed for reproducibility in logic, though instances have hidden shifts
        
        points = []
        values = []
        
        # Center
        x_center = np.zeros(self.dim)
        f_center = func(x_center)
        points.append(x_center)
        values.append(f_center)
        
        # Random points
        for _ in range(n_global):
            x_rand = rng.uniform(-5, 5, self.dim)
            f_rand = func(x_rand)
            points.append(x_rand)
            values.append(f_rand)
            
        # Find top 3 best points
        # Sort indices by value
        sorted_indices = np.argsort(values)
        top_indices = sorted_indices[:3]
        
        best_x = points[top_indices[0]]
        best_f = values[top_indices[0]]
        
        calls_used = len(points)
        remaining_budget = self.budget - calls_used
        
        # Phase 2: Local Refinement with Nelder-Mead
        # Split remaining budget among the top 3 candidates
        # If we have less than 50 calls left, just use (1+1)-ES or return
        if remaining_budget > 30:
            budget_per_candidate = remaining_budget // 3
            # Ensure at least 10 calls per candidate if possible
            if budget_per_candidate < 10:
                budget_per_candidate = 10
            else:
                # Adjust to use all budget
                # 2 * budget_per_candidate + 1 * (remaining - 2*bp)
                # Simpler: just give each 1/3, and let the last one take the rest?
                # Or just give each 1/3 and ignore the small remainder.
                pass
            
            for idx in top_indices:
                if time.time() > deadline - 5:
                    break
                
                x0 = points[idx]
                f0 = values[idx]
                
                # Wrap function to ensure we don't exceed budget?
                # scipy's maxfev handles this.
                try:
                    res = minimize(
                        func, 
                        x0, 
                        method='Nelder-Mead', 
                        options={
                            'maxfev': budget_per_candidate,
                            'xatol': 1e-8,
                            'fatol': 1e-8,
                            'adaptive': True
                        }
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except Exception:
                    pass

        return best_x, best_f
