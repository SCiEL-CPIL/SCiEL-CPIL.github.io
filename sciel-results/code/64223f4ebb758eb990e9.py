import numpy as np
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        
        # Phase 1: (1+1)-ES
        # Reserve 50 calls for polishing
        polish_budget = 50
        es_budget = self.budget - polish_budget
        
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        
        sigma = 2.0
        
        calls_used = 1
        
        for _ in range(es_budget - 1):
            # Generate candidate
            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            calls_used += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                sigma *= 0.87
                
            # Optional: If stuck, restart?
            # For now, keep it simple.
            
        # Phase 2: Nelder-Mead Polish
        if polish_budget > 0:
            # Wrapper to count calls and enforce budget
            calls_remaining = [polish_budget]
            
            def wrapped_func(x_eval):
                if calls_remaining[0] <= 0:
                    return best_fx # Return best known to stop
                calls_remaining[0] -= 1
                return func(x_eval)
            
            # Initialize simplex
            # Use a small perturbation
            x0 = best_x.copy()
            # If best_fx is very good, use smaller steps
            # Default NM uses x0 + 0.05 * (x0 - x_i) or similar
            # We can control initial simplex size via xatol or options
            try:
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='Nelder-Mead', 
                    options={'maxiter': polish_budget, 'xatol': 1e-10, 'fatol': 1e-10}
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_fx
