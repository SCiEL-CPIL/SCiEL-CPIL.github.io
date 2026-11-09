import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.elapsed_time_start = None
        # 280s is a safe margin for 300s limit
        self.max_time = 280.0 

    def __call__(self, func):
        self.elapsed_time_start = time.time()
        dim = self.dim
        budget = self.budget
        
        # State tracking
        best_x = np.zeros(dim)
        best_f = np.inf
        eval_count = 0
        
        # Random number generator
        rng = np.random.default_rng(12345)

        def wrapped_func(x):
            nonlocal eval_count, best_f, best_x
            # Time guard
            if time.time() - self.elapsed_time_start > self.max_time:
                raise TimeoutError("Time limit exceeded")
            
            eval_count += 1
            if eval_count > budget:
                raise ValueError("Budget exceeded")
            
            x = np.asarray(x).flatten()
            # Ensure x is within bounds for the function call, 
            # although func clips, it's good practice to pass valid inputs if possible.
            # But func handles clipping.
            try:
                f_val = func(x)
            except Exception:
                # If func fails, return a large penalty
                return 1e100
            
            if np.isfinite(f_val) and f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            
            return f_val

        # Helper to run NM with a specific start and maxfev
        def run_nm(x_start, maxfev):
            nonlocal eval_count
            if eval_count >= budget:
                return
            if time.time() - self.elapsed_time_start > self.max_time:
                return
            
            # Clip start point to box
            x_start = np.clip(x_start, -5, 5)
            
            try:
                minimize(
                    wrapped_func, 
                    x_start, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': 100, # NM iterations, not evals
                        'maxfev': maxfev,
                        'xatol': 1e-10,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
            except Exception:
                # Catch any exceptions from minimize (e.g. from wrapped_func timeout/budget)
                pass

        # Phase 1: Global search / Initial convergence
        # Start from center
        run_nm(np.zeros(dim), 200)
        
        # Phase 2: Refinement from best found
        # Use a smaller maxfev to polish
        remaining = budget - eval_count
        if remaining > 20:
            run_nm(best_x, min(100, remaining))
            
        # Phase 3: Diversification / Second attempt
        # Try a random start if we have budget
        remaining = budget - eval_count
        if remaining > 50:
            x_rand = rng.uniform(-5, 5, dim)
            run_nm(x_rand, min(100, remaining))
            
        # Phase 4: Final Polish
        remaining = budget - eval_count
        if remaining > 10:
            run_nm(best_x, min(50, remaining))

        # Fallback if no evaluations were made (should not happen)
        if eval_count == 0:
             x = np.zeros(dim)
             try:
                 f = func(x)
                 best_x = x
                 best_f = f
             except:
                 pass

        return best_x, best_f
