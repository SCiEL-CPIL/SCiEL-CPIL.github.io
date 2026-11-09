"""Optimizer for BBOB-style f5 (Linear slope)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        best_x = np.zeros(self.dim)
        best_f = np.inf
        
        # Track total calls to ensure we don't exceed budget significantly
        # though the deadline guard is the primary safety.
        total_calls = 0
        
        rng = np.random.default_rng(0)
        
        # Generate starting points
        # 1. Origin
        # 2. Random points
        starting_points = [np.zeros(self.dim)]
        for _ in range(4):
            starting_points.append(rng.uniform(-5, 5, self.dim))
        
        for x0 in starting_points:
            if time.time() > deadline:
                break
            if total_calls >= self.budget:
                break
                
            x0 = np.clip(x0, -5, 5)
            
            # Construct initial simplex with step size 1.0
            # x0, x0 + 1*e1, x0 + 1*e2, ...
            simplex = [x0]
            for i in range(self.dim):
                xi = x0.copy()
                xi[i] += 1.0
                simplex.append(xi)
            initial_simplex = np.array(simplex)
            
            calls = 0
            def obj(x):
                nonlocal calls, total_calls
                if time.time() > deadline:
                    raise Exception("Deadline exceeded")
                if total_calls >= self.budget:
                    raise Exception("Budget exceeded")
                
                x_clipped = np.clip(x, -5, 5)
                calls += 1
                total_calls += 1
                return func(x_clipped)
            
            try:
                res = minimize(obj, x0, method='Nelder-Mead', 
                               options={'maxiter': 200, 'fatol': 1e-12, 'xatol': 1e-12, 'adaptive': True},
                               initial_simplex=initial_simplex)
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
                    
            except Exception:
                pass

        # Ensure we have a valid best_x
        if best_f == np.inf:
             best_x = np.zeros(self.dim)
             best_f = func(best_x)
             
        return best_x, best_f
# EVOLVE-BLOCK-END
