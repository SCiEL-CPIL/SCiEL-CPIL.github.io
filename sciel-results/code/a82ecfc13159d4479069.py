import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0 # Safety margin
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Phase 1: LHS Initialization
        # Use 10% of budget for LHS, min 20, max 50
        n_lhs = min(50, max(20, budget // 10))
        if calls_made + n_lhs > budget:
            n_lhs = budget - calls_made
            
        # Generate LHS points
        lhs_points = np.zeros((n_lhs, dim))
        for d in range(dim):
            perm = rng.permutation(n_lhs)
            # Uniform in [-5, 5]
            lhs_points[:, d] = -5 + 10 * (perm + rng.random(n_lhs)) / n_lhs
            
        for i in range(n_lhs):
            if time.time() > deadline:
                break
            x = lhs_points[i]
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        # Phase 2: (1+1)-ES
        # Use 70% of remaining budget? Or fixed ratio?
        # Let's say we reserve last 100 calls for NM if possible, or 20% of total.
        nm_budget = min(100, budget // 5) # Reserve up to 100 calls for NM
        es_budget = budget - calls_made - nm_budget
        if es_budget < 50: # If not enough for ES, give more to ES or skip NM
             nm_budget = 0
             es_budget = budget - calls_made
             
        if best_x is not None and es_budget > 0:
            x = best_x.copy()
            fx = best_f
            sigma = 1.0
            success_rate = 0.5
            window = 10
            successes = 0
            trial_count = 0
            
            for _ in range(es_budget):
                if time.time() > deadline:
                    break
                
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    pass # keep x, fx
                
                trial_count += 1
                if trial_count >= window:
                    success_rate = successes / window
                    if success_rate > 0.3:
                        sigma *= 1.2
                    elif success_rate < 0.1:
                        sigma *= 0.8
                    successes = 0
                    trial_count = 0
                    
                sigma = np.clip(sigma, 1e-4, 5.0)
                
        # Phase 3: Nelder-Mead Polish
        if best_x is not None and nm_budget > 0 and calls_made < budget:
            remaining = budget - calls_made
            # NM uses dim+1 evaluations per iteration roughly.
            # We can set maxfev to remaining.
            try:
                res = minimize(
                    func, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={'maxiter': 1000, 'maxfev': remaining, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                # Check if result is better
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                pass # If NM fails, we still have the best from ES

        return best_x, best_f
