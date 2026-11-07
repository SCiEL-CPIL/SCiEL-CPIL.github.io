import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        
        n = self.dim
        total_budget = self.budget
        
        # Phase 1: (1+1)-ES for global exploration
        # Use 50% of budget for exploration
        exp_budget = int(total_budget * 0.5)
        
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, n)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        sigma = 2.0
        calls = 1
        
        while calls < exp_budget:
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                sigma *= 0.87
        
        # Phase 2: Local Optimization with Powell
        # Use remaining budget
        rem_budget = total_budget - calls
        
        if rem_budget > 0:
            state = {'calls': 0, 'best_x': best_x, 'best_f': best_f}
            
            def wrapped_func(x_eval):
                if time.time() > deadline:
                    raise ValueError("Deadline exceeded")
                state['calls'] += 1
                if state['calls'] > rem_budget:
                    raise ValueError("Budget exceeded")
                val = func(x_eval)
                if val < state['best_f']:
                    state['best_f'] = val
                    state['best_x'] = x_eval.copy()
                return val
            
            try:
                # Powell is often better for Rosenbrock than Nelder-Mead
                res = minimize(wrapped_func, best_x, method='Powell', 
                               options={'maxiter': 100, 'maxfev': rem_budget, 'xatol': 1e-8, 'fatol': 1e-10})
            except ValueError:
                pass
            except Exception:
                pass

        return best_x, best_f
