import numpy as np
import time
import itertools
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        # 1. Coarse Grid Search
        # 3 points per dimension in [-4, 4]
        grid_vals = np.array([-4.0, 0.0, 4.0])
        grid_points = list(itertools.product(grid_vals, repeat=self.dim))
        
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        for p in grid_points:
            if time.time() > deadline or calls_made >= self.budget:
                break
            x = np.array(p)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        if best_x is None:
            # Fallback: center
            best_x = np.zeros(self.dim)
            best_f = func(best_x)
            calls_made += 1

        remaining_budget = self.budget - calls_made
        if remaining_budget <= 0 or time.time() > deadline:
            return best_x, best_f

        # 2. Local Refinement with Nelder-Mead
        # Construct a larger initial simplex to help escape local minima or flat areas
        # Default simplex is small. We use step size 0.5
        step = 0.5
        initial_simplex = np.zeros((self.dim + 1, self.dim))
        initial_simplex[0] = best_x
        for i in range(self.dim):
            initial_simplex[i + 1] = best_x
            initial_simplex[i + 1, i] += step
            
        # Check if any point in simplex is out of bounds?
        # func clips, so it's fine, but large steps might be wasted.
        # 0.5 is reasonable.

        def wrapped_func(x):
            nonlocal calls_made
            if time.time() > deadline or calls_made >= self.budget:
                return float('inf')
            fx = func(x)
            calls_made += 1
            return fx

        try:
            res = minimize(wrapped_func, best_x, method='Nelder-Mead', 
                           options={
                               'maxiter': 1000, 
                               'maxfev': remaining_budget, 
                               'xatol': 1e-8, 
                               'fatol': 1e-8,
                               'initial_simplex': initial_simplex
                           })
            
            if res.fun < best_f and np.all(np.isfinite(res.fun)):
                best_x = res.x
                best_f = res.fun
                
        except Exception:
            pass
            
        return best_x, best_f
