import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        global_best_x = None
        global_best_f = float('inf')
        evals_used = 0

        def check_deadline():
            return time.time() > deadline

        def safe_func(x):
            nonlocal evals_used
            if evals_used >= budget:
                raise ValueError("Budget exceeded")
            if check_deadline():
                raise ValueError("Time exceeded")
            
            # Clip to box
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            evals_used += 1
            return fx

        # Phase 1: Initial Exploration
        # Try origin
        x0 = np.zeros(dim)
        try:
            fx0 = safe_func(x0)
            global_best_f = fx0
            global_best_x = x0.copy()
        except:
            pass

        # Try a few random points to find a better basin
        # Rastrigin has many minima. Origin is often near a local min, but not always the global one.
        # Global min is in [-4, 4]^5.
        n_initial = 10
        for _ in range(n_initial):
            if evals_used >= budget or check_deadline():
                break
            x_try = rng.uniform(-4, 4, dim)
            try:
                fx_try = safe_func(x_try)
                if fx_try < global_best_f:
                    global_best_f = fx_try
                    global_best_x = x_try.copy()
            except:
                break

        # Phase 2: Local Optimization with Restarts
        # Use Nelder-Mead or Powell. Nelder-Mead is robust.
        # We will run minimize from the current best, and if it doesn't improve much, restart from a new random point.
        
        max_local_evals = 50 # Allow up to 50 evals per local run to keep budget for restarts
        min_improvement = 1e-5

        while evals_used < budget and not check_deadline():
            # Determine starting point for local search
            # 50% chance to restart from current best (refinement)
            # 50% chance to restart from a new random point (exploration)
            # Since we already did initial exploration, we can bias more towards refinement of the best.
            # But Rastrigin needs exploration. Let's alternate or use a ratio.
            
            if rng.random() < 0.5 and global_best_x is not None:
                x_start = global_best_x.copy()
            else:
                x_start = rng.uniform(-4, 4, dim)
                # Evaluate starting point if not already done? 
                # The minimize function will evaluate it.
                # But we need to know its value to decide if it's worth running.
                # Let's just let minimize run.
            
            # Estimate remaining budget
            remaining = budget - evals_used
            if remaining < 5:
                break
                
            # Set maxiter for scipy
            # Nelder-Mead uses roughly (dim+1) evals per iteration initially, then more.
            # Let's cap the number of iterations to use at most 'max_local_evals' evaluations.
            # A rough estimate: maxiter = max(10, max_local_evals // 5)
            max_iter = max(10, min(50, remaining // 5))
            
            try:
                res = minimize(
                    safe_func, 
                    x_start, 
                    method='Nelder-Mead',
                    options={
                        'maxiter': max_iter,
                        'xatol': 1e-4,
                        'fatol': 1e-4,
                        'disp': False
                    }
                )
                
                if res.fun < global_best_f:
                    global_best_f = res.fun
                    global_best_x = res.x.copy()
                    
            except (ValueError, Exception):
                # Budget or time exceeded inside minimize
                pass

        return global_best_x, global_best_f
