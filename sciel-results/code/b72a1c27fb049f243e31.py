import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(12345)
        
        # Phase 1: Global Exploration (20% of budget)
        global_budget = max(20, budget // 5)
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls = 1
        
        for _ in range(global_budget - 1):
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
        
        # Phase 2: Local Exploitation
        if calls < budget and time.time() < deadline:
            remaining_budget = budget - calls
            options = {
                'maxiter': remaining_budget,
                'xatol': 1e-10,
                'fatol': 1e-12,
                'adaptive': True
            }
            
            try:
                res = minimize(func, best_x, method='Nelder-Mead', options=options)
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                calls += res.nfev
            except Exception:
                # Fallback to simple local search if Nelder-Mead fails
                # Simple (1+1)-ES for remaining budget
                x_curr = best_x.copy()
                f_curr = best_f
                sigma = 0.5
                
                while calls < budget:
                    if time.time() > deadline:
                        break
                    
                    y = x_curr + sigma * rng.standard_normal(dim)
                    y = np.clip(y, -5.0, 5.0)
                    fy = func(y)
                    calls += 1
                    
                    if fy < f_curr:
                        x_curr = y
                        f_curr = fy
                        if fy < best_f:
                            best_f = fy
                            best_x = y.copy()
                        sigma *= 1.05
                    else:
                        sigma *= 0.95
                        
                        if sigma < 1e-6:
                            break

        return best_x, best_f
