import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')
        self.start_time = None
        self.deadline = None

    def __call__(self, func):
        self.start_time = time.time()
        self.deadline = self.start_time + 280  # Safety margin

        def safe_func(x):
            if self.calls >= self.budget:
                raise ValueError("Budget exceeded")
            if time.time() > self.deadline:
                raise TimeoutError("Time limit exceeded")
            
            # Ensure x is a numpy array of correct shape
            x = np.asarray(x, dtype=float)
            if x.shape != (self.dim,):
                x = x.reshape(self.dim)
            
            self.calls += 1
            f_val = func(x)
            
            if f_val < self.best_f:
                self.best_f = f_val
                self.best_x = x.copy()
                
            return f_val

        rng = np.random.default_rng()

        # Phase 1: Random Initialization
        # Use a portion of the budget for global exploration
        # 50 calls is a good balance for dim=5
        n_init = min(50, self.budget)
        
        init_points = []
        init_vals = []
        
        for _ in range(n_init):
            if self.calls >= self.budget or time.time() > self.deadline:
                break
            x = rng.uniform(-5, 5, self.dim)
            try:
                f = safe_func(x)
                init_points.append(x)
                init_vals.append(f)
            except (ValueError, TimeoutError):
                break
        
        if not init_points:
            # Fallback if no calls were made (should not happen)
            x = np.zeros(self.dim)
            try:
                safe_func(x)
            except:
                pass
            return self.best_x, self.best_f

        # Phase 2: Local Refinement using Nelder-Mead
        # Sort initial points by value
        sorted_indices = np.argsort(init_vals)
        # Take the top 5 starting points
        n_restarts = min(5, len(sorted_indices))
        
        # Nelder-Mead options
        # We want it to stop if it doesn't improve significantly or if budget runs out
        # maxiter is high, but we rely on the budget wrapper to stop it
        nm_options = {
            'maxiter': 200,
            'xatol': 1e-6,
            'fatol': 1e-6,
            'adaptive': True
        }
        
        for i in range(n_restarts):
            if self.calls >= self.budget or time.time() > self.deadline:
                break
                
            x0 = init_points[sorted_indices[i]]
            
            try:
                # Run Nelder-Mead
                # The wrapper will raise an exception if budget is exceeded
                res = scipy.optimize.minimize(
                    safe_func, 
                    x0, 
                    method='Nelder-Mead', 
                    options=nm_options
                )
            except (ValueError, TimeoutError):
                # Budget or time exceeded, stop further restarts
                break
            except Exception as e:
                # Other errors, try next restart
                continue
            
            # If we get here, the minimization finished naturally (converged)
            # Check if the result is better (it should be, but safe_func tracks it)
            # No need to explicitly update best_x/f here as safe_func did it

        return self.best_x, self.best_f
