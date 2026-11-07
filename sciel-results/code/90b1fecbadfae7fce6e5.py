import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = 280.0  # Safety margin before 300s limit
        n = self.dim
        
        best_x = np.zeros(n)
        best_f = np.inf
        total_evals = 0

        def safe_func(x):
            nonlocal total_evals
            # Check time
            if time.time() - start_time > deadline:
                raise Exception("Time limit exceeded")
            
            # Validate input
            if not isinstance(x, np.ndarray) or x.shape != (n,) or not np.all(np.isfinite(x)):
                return 1e10
                
            total_evals += 1
            if total_evals > self.budget:
                return 1e10
                
            try:
                val = float(func(x))
                if not np.isfinite(val):
                    return 1e10
                return val
            except Exception:
                return 1e10

        # Multi-start Nelder-Mead
        # Nelder-Mead is robust and doesn't require Hessian/Covariance updates
        num_starts = 8
        
        # 1. Center start
        starts = [np.zeros(n)]
        # 2. Random starts
        for _ in range(num_starts - 1):
            starts.append(np.random.uniform(-5, 5, n))

        for x0 in starts:
            if time.time() - start_time > deadline:
                break
            if total_evals >= self.budget:
                break
            
            # Estimate max iterations based on remaining budget
            # NM uses ~n+1 evals per iteration
            remaining_budget = self.budget - total_evals
            max_iter = max(10, remaining_budget // (n + 1))
            
            try:
                res = minimize(safe_func, x0, method='Nelder-Mead', 
                               options={'maxiter': max_iter, 'xatol': 1e-5, 'fatol': 1e-7})
                
                if res.x is not None and np.all(np.isfinite(res.x)):
                    # Update global best if this local minimum is better
                    # Note: res.fun might be from the last call, which is safe
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                        
            except Exception:
                # If time limit or other error, stop
                break

        return best_x, best_f
