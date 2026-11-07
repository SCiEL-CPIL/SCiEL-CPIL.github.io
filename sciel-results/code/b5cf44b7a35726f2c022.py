import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Deadline guard
        start_time = time.time()
        deadline = start_time + 280.0
        
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        def count_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            calls_made += 1
            x = np.clip(x, -5, 5)
            f = func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Strategy: Grid-based multi-start + Nelder-Mead refinement
        # 1. Evaluate a sparse 3^5 grid (243 points)
        # 2. Select top 5 points
        # 3. Run Nelder-Mead from each of these 5 points with remaining budget
        
        # Step 1: Grid evaluation
        # Grid points: -3, 0, 3 (covers most of [-5,5] reasonably well for 5D)
        # 3^5 = 243 points.
        grid_values = [-3.0, 0.0, 3.0]
        grid_points = []
        
        for v1 in grid_values:
            for v2 in grid_values:
                for v3 in grid_values:
                    for v4 in grid_values:
                        for v5 in grid_values:
                            grid_points.append([v1, v2, v3, v4, v5])
        
        # Evaluate grid
        grid_results = []
        for pt in grid_points:
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break
            x = np.array(pt)
            f = count_func(x)
            grid_results.append((f, x))
        
        # If we didn't finish the grid, just use what we have
        if not grid_results:
            # Fallback: single point at 0
            if calls_made < budget:
                x0 = np.zeros(dim)
                f0 = count_func(x0)
                grid_results.append((f0, x0))
        
        # Step 2: Select top 5
        grid_results.sort(key=lambda x: x[0])
        top_k = min(5, len(grid_results))
        top_points = [x for _, x in grid_results[:top_k]]
        
        # Step 3: Nelder-Mead refinement
        # Remaining budget
        remaining_budget = budget - calls_made
        
        if remaining_budget > 0 and top_points:
            # Distribute remaining budget among top points
            # Simple distribution: equal share
            budget_per_point = remaining_budget // top_k
            # Ensure at least 10 calls per point if possible
            if budget_per_point < 10:
                budget_per_point = 10
            
            for i, x0 in enumerate(top_points):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                
                # Allocate a specific sub-budget for this point
                # We need a way to stop Nelder-Mead after N calls.
                # We can use a wrapper that raises an exception or returns a specific value?
                # SciPy minimize doesn't support custom call limits directly easily.
                # We can use a counter and a flag.
                
                sub_calls = [0]
                sub_limit = budget_per_point
                
                def sub_count_func(x):
                    nonlocal calls_made, best_x, best_f
                    if sub_calls[0] >= sub_limit:
                        raise ValueError("Budget exhausted")
                    if calls_made >= budget:
                        raise ValueError("Global budget exhausted")
                    calls_made += 1
                    sub_calls[0] += 1
                    x = np.clip(x, -5, 5)
                    f = func(x)
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    return f
                
                try:
                    # Nelder-Mead parameters
                    # initial_simplex: default is based on x0
                    # maxiter: set high, rely on call limit
                    res = minimize(sub_count_func, x0, method='Nelder-Mead', 
                                   options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-6})
                except ValueError:
                    pass
                except Exception:
                    pass
        
        return best_x, best_f
