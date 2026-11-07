import numpy as np
import time
import math
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        # 330s wall-clock limit. 290s deadline leaves 40s margin.
        deadline = start_time + 290.0

        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        def wrapper(x):
            nonlocal calls_made, best_x, best_f
            # Check time
            if time.time() > deadline:
                return best_f
            # Check budget
            if calls_made >= budget:
                return best_f
            
            try:
                f_val = func(np.array(x))
            except Exception:
                return best_f
            
            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = np.array(x)
            return f_val

        # Strategy: Grid-based multi-start followed by local polish.
        # 1. Evaluate a 3^5 grid (243 points) to find basins.
        # 2. Select top 5 points.
        # 3. Run Nelder-Mead from each, using the remaining budget (~257 calls).
        
        # Generate 3^5 grid points in [-5, 5]
        # Levels: -5, 0, 5
        # Actually, for a 5D box, 3 levels per dimension gives 3^5 = 243 points.
        # This is a coarse global search.
        # To be more effective, maybe 3 levels is too coarse? 
        # The budget is 500. 243 for grid leaves 257 for local.
        # 257 / 5 starts = ~51 calls per Nelder-Mead.
        # Nelder-Mead in 5D needs ~5-10 iterations to converge, each iteration ~5-6 calls.
        # So 50 calls is enough for a few iterations.
        
        # Let's use a slightly finer grid if possible?
        # 3^5 = 243.
        # 2^5 = 32 (corners). Too few.
        # Maybe 3^5 is good.
        
        grid_levels = np.array([-5.0, 0.0, 5.0])
        grid_points = np.array(np.meshgrid(*([grid_levels]*dim))).T.reshape(-1, dim)
        
        # Evaluate grid
        grid_fvals = []
        for i in range(len(grid_points)):
            if time.time() > deadline or calls_made >= budget:
                break
            x = grid_points[i]
            f = wrapper(x)
            grid_fvals.append(f)
        
        if len(grid_fvals) < 5:
            # Not enough budget or time, return best found
            return best_x, best_f

        # Select top 5 unique points (though grid points are unique)
        # Sort by fitness
        sorted_indices = np.argsort(grid_fvals)
        top_k = min(5, len(sorted_indices))
        top_indices = sorted_indices[:top_k]
        
        # If we ran out of budget during grid, top_indices might be incomplete
        # but we have at least some.
        
        # Phase 2: Local Optimization with Nelder-Mead
        # We have remaining_budget calls.
        # Distribute them among the top 5 starts.
        # Nelder-Mead options: maxiter, xatol, fatol.
        # We want to use as many calls as possible but stay within budget.
        # The wrapper enforces the budget.
        
        for idx in top_indices:
            if time.time() > deadline - 5.0 or calls_made >= budget:
                break
            
            x0 = grid_points[idx]
            
            try:
                # Nelder-Mead is robust and doesn't need gradients.
                # maxiter: limit iterations to avoid hanging.
                # With ~50 calls left, and ~5 calls/iter, maxiter=10 is safe.
                # But if we have more budget, we can allow more.
                # Let's just let it run until budget/time is hit by wrapper.
                # However, Nelder-Mead might keep going if it thinks it's improving.
                # Setting a maxiter is safer.
                # Remaining budget: budget - calls_made
                # Approx calls per iter: dim + 1 = 6.
                # maxiter = (budget - calls_made) // 6 + 1
                max_iter_est = max(5, (budget - calls_made) // 6)
                
                res = minimize(
                    wrapper,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter_est,
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                pass

        return best_x, best_f
