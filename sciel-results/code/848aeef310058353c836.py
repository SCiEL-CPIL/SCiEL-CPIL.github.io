import numpy as np
import time
from scipy.optimize import minimize
import itertools

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        all_points = [] # List of (x, f)

        def safe_call(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget or time.time() > deadline:
                return np.inf
            # Ensure x is a 1D numpy array of floats
            x = np.array(x, dtype=float).flatten()
            val = func(x)
            calls_made += 1
            all_points.append((x, val))
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Coarse Grid
        # 3 points per dimension
        # We use centers of 3 equal intervals in [-5, 5]
        # Intervals: [-5, -1.66], [-1.66, 1.66], [1.66, 5]
        # Centers: -3.33, 0, 3.33
        coarse_vals = [-3.333333, 0.0, 3.333333]
        coarse_points = np.array(list(itertools.product(coarse_vals, repeat=dim)))
        
        for x in coarse_points:
            safe_call(x)
            if calls_made >= budget or time.time() > deadline:
                return best_x, best_f
        
        # Sort by fitness to find top candidates
        all_points.sort(key=lambda item: item[1])
        
        # Phase 2: Medium Grid around top K
        # Refine the best K cells
        K = 10
        top_k = all_points[:K]
        
        # Subdivide the cell containing the coarse point into 2^dim sub-cells
        # Cell width is approx 3.33. Half-width 1.66.
        # Sub-cell width 1.66. Half-width 0.83.
        # Offsets from center: -0.83, 0.83
        offsets = [-0.833333, 0.833333]
        
        for x_center, f_center in top_k:
            if calls_made >= budget or time.time() > deadline:
                break
            
            # Generate 2^dim sub-points
            sub_offsets = np.array(list(itertools.product(offsets, repeat=dim)))
            for offset in sub_offsets:
                x_new = x_center + offset
                # Clip to bounds to be safe
                x_new = np.clip(x_new, -5, 5)
                safe_call(x_new)
                if calls_made >= budget or time.time() > deadline:
                    break
        
        # Phase 3: Local Refinement
        # Re-sort all points to find the global best so far
        all_points.sort(key=lambda item: item[1])
        
        M = 5
        top_m = all_points[:M]
        
        remaining_budget = budget - calls_made
        if remaining_budget > 0:
            # Allocate remaining budget to local searches
            # Ensure at least some budget per start
            budget_per_start = max(10, remaining_budget // M)
            
            for x0, f0 in top_m:
                if calls_made >= budget or time.time() > deadline:
                    break
                
                def wrapped_func(x):
                    if time.time() > deadline or calls_made >= budget:
                        return best_f
                    return safe_call(x)
                
                try:
                    minimize(
                        wrapped_func,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 100, 'maxfev': budget_per_start, 'xatol': 1e-6, 'fatol': 1e-6}
                    )
                except Exception:
                    pass
        
        return best_x, best_f
