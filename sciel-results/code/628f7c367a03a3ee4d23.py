import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
        def __init__(self, budget, dim):
            self.budget = budget
            self.dim = dim

        def __call__(self, func):
            start_time = time.time()
            # 300s limit, keep 20s buffer for safety
            time_limit = 300 - 20
            
            calls_made = 0
            best_x = np.zeros(self.dim)
            best_f = float('inf')
            
            def wrapper(x):
                nonlocal calls_made, best_x, best_f
                # Time check
                if time.time() - start_time > time_limit:
                    return best_f
                # Budget check
                if calls_made >= self.budget:
                    return best_f
                
                calls_made += 1
                try:
                    f_val = func(x)
                except Exception:
                    return best_f
                
                if f_val < best_f:
                    best_f = f_val
                    best_x = np.array(x)
                return f_val

            # Phase 1: Grid-based global search
            # Evaluate a 3^5 grid to find top basins
            grid_points = []
            # 3 levels per dimension: -3, 0, 3 (within [-5,5])
            levels = [-3.0, 0.0, 3.0]
            
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        for l in range(3):
                            for m in range(3):
                                grid_points.append(np.array([levels[i], levels[j], levels[k], levels[l], levels[m]]))
            
            # Evaluate all grid points
            grid_scores = []
            for pt in grid_points:
                if calls_made >= self.budget or time.time() - start_time > time_limit:
                    break
                f_val = wrapper(pt)
                grid_scores.append((f_val, pt))
            
            # Sort by fitness and pick top 5 distinct basins
            grid_scores.sort(key=lambda x: x[0])
            top_points = [pt for _, pt in grid_scores[:5]]
            
            # Phase 2: Local polish from each top point
            for start_pt in top_points:
                if calls_made >= self.budget or time.time() - start_time > time_limit:
                    break
                
                try:
                    res_nm = minimize(
                        wrapper,
                        start_pt,
                        method='Nelder-Mead',
                        options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                    )
                    if res_nm.fun < best_f:
                        best_f = res_nm.fun
                        best_x = np.array(res_nm.x)
                except Exception:
                    pass
            
            # Phase 3: Final high-precision polish from best point
            if calls_made < self.budget and time.time() - start_time < time_limit:
                try:
                    res_lbfgsb = minimize(
                        wrapper,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * self.dim,
                        options={'maxiter': 100, 'gtol': 1e-12, 'ftol': 1e-12}
                    )
                    if res_lbfgsb.fun < best_f:
                        best_f = res_lbfgsb.fun
                        best_x = np.array(res_lbfgsb.x)
                except Exception:
                    pass
            
            return best_x, best_f
