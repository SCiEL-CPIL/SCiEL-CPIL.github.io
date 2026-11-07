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
        time_limit = 280  # 20s buffer before 300s kill
        
        calls_made = 0
        best_x = None
        best_f = float('inf')
        N = self.dim
        budget = self.budget

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if time.time() - start_time > time_limit:
                raise TimeoutError()
            if calls_made >= self.budget:
                raise RuntimeError()
            
            calls_made += 1
            try:
                f_val = func(x)
            except Exception:
                return best_f if best_f != float('inf') else 0.0
            
            if f_val < best_f:
                best_f = f_val
                best_x = np.array(x)
            return f_val

        # Phase 1: Sparse Grid Search for Global Basin Identification
        # 3^5 = 243 points covering [-5, 5]^5 with levels at -5, 0, 5
        grid_points = []
        if budget >= 250:
            levels = [-5.0, 0.0, 5.0]
            for combo in itertools.product(levels, repeat=N):
                grid_points.append(np.array(combo))
        else:
            grid_points = [np.zeros(N)]

        # Evaluate grid
        grid_results = []
        try:
            for x in grid_points:
                f_val = evaluate(x)
                grid_results.append((f_val, x))
        except (TimeoutError, RuntimeError):
            pass

        # Sort grid results by fitness
        grid_results.sort(key=lambda x: x[0])
        
        # Select top K basins for local optimization
        K = min(5, len(grid_results))
        top_basins = [x for _, x in grid_results[:K]]
        
        # If grid search used all budget, we are done.
        if calls_made >= budget or not top_basins:
            return best_x, best_f

        # Phase 2: Local Polish with L-BFGS-B
        # L-BFGS-B is much more effective for Rosenbrock than Nelder-Mead
        # because it uses gradient information and handles the valley structure well.
        
        def lbf_wrapper(x):
            nonlocal calls_made, best_x, best_f
            if time.time() - start_time > time_limit:
                raise TimeoutError()
            if calls_made >= self.budget:
                raise RuntimeError()
            
            calls_made += 1
            try:
                f_val = func(x)
            except Exception:
                return best_f if best_f != float('inf') else 0.0
            
            if f_val < best_f:
                best_f = f_val
                best_x = np.array(x)
            return f_val

        # Iterate through top basins, allocating more budget to better ones
        for i, start_x in enumerate(top_basins):
            if calls_made >= budget or time.time() - start_time > time_limit:
                break
            
            # Allocate more iterations to the best basins
            if i == 0:
                max_iter = 300
            elif i == 1:
                max_iter = 200
            else:
                max_iter = 100
            
            try:
                res = minimize(
                    lbf_wrapper, 
                    start_x, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * N,
                    options={
                        'maxiter': max_iter,
                        'ftol': 1e-12,
                        'gtol': 1e-10,
                        'maxls': 30
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.array(res.x)
                    
            except (TimeoutError, RuntimeError):
                break
            except Exception:
                pass

        # Phase 3: Final high-precision polish on the best point found
        if calls_made < budget and time.time() - start_time < time_limit:
            try:
                res = minimize(
                    lbf_wrapper, 
                    best_x, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * N,
                    options={
                        'maxiter': 500,
                        'ftol': 1e-14,
                        'gtol': 1e-12,
                        'maxls': 50
                    }
                )
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.array(res.x)
            except (TimeoutError, RuntimeError):
                pass
            except Exception:
                pass

        return best_x, best_f
