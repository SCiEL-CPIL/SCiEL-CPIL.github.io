import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin
        
        dim = self.dim
        budget = self.budget
        
        # 1. Initialization with a set of spread points
        # Center and axis-aligned points
        points = [np.zeros(dim)]
        for i in range(dim):
            for sign in [-1, 1]:
                p = np.zeros(dim)
                p[i] = sign * 3.0
                points.append(p)
        
        # Evaluate initial points
        f_values = []
        for p in points:
            f_val = func(p)
            f_values.append(f_val)
            
        # Find best
        best_idx = np.argmin(f_values)
        x_best = points[best_idx].copy()
        f_best = f_values[best_idx]
        
        best_x = x_best
        best_f = f_best
        eval_count = len(points)
        
        def wrapped_func(x):
            nonlocal eval_count, best_f, best_x
            if time.time() > deadline:
                raise Exception("Time limit exceeded")
            if eval_count >= budget:
                raise Exception("Budget exceeded")
            eval_count += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # 2. Optimization with Powell's Method
        # Powell's method is effective for functions with ridges or separable structure.
        try:
            remaining_budget = budget - eval_count
            # Reserve a few calls for potential polish or safety
            max_fev = max(10, remaining_budget - 20)
            
            res = minimize(wrapped_func, x_best, method='Powell', 
                           options={'maxiter': 2000, 'maxfev': max_fev, 'ftol': 1e-12, 'xtol': 1e-12})
            
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
                
        except Exception as e:
            # If it fails, we keep the best found so far
            pass

        # 3. Polish with Nelder-Mead if budget remains
        if time.time() < deadline - 5 and eval_count < budget - 10:
            try:
                remaining = budget - eval_count - 5
                if remaining > 10:
                    res2 = minimize(wrapped_func, best_x, method='Nelder-Mead', 
                           options={'maxiter': 100, 'maxfev': remaining, 'xatol': 1e-10, 'fatol': 1e-10})
                    if res2.fun < best_f:
                        best_f = res2.fun
                        best_x = res2.x.copy()
            except Exception:
                pass

        return best_x, best_f
