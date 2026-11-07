"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Global scout + Powell's method for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s limit, use 320s deadline
        deadline = start_time + 320
        
        best_x = None
        best_f = float('inf')
        eval_count = 0
        dim = self.dim
        budget = self.budget

        def wrapped_func(x):
            nonlocal eval_count, best_x, best_f
            # Hard stop if we exceed budget or time
            if eval_count >= budget:
                raise Exception("Budget exceeded")
            if time.time() > deadline:
                raise Exception("Time exceeded")
            
            x = np.asarray(x, dtype=float)
            if x.shape != (dim,):
                x = np.clip(x, -5, 5)
            
            fx = func(x)
            eval_count += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            return fx

        # Phase 1: Global Scout
        # Rosenbrock is tricky because the minimum can be shifted.
        # A coarse grid helps locate the general region.
        # 5D space. Let's try a grid of 3 points per dimension? 3^5 = 243 points.
        # That's a lot of evals. 
        # Maybe 2 points per dimension? 2^5 = 32 points.
        # Or a mix: Center, corners, and mid-points.
        
        # Let's use a set of strategic points:
        # 1. Origin (0,0,0,0,0)
        # 2. Standard min (1,1,1,1,1)
        # 3. -Standard min (-1,-1,-1,-1,-1)
        # 4. Box corners (±5, ±5, ±5, ±5, ±5) -> 32 points
        # 5. Mid-points (±2.5, ...) -> 32 points
        
        scout_points = []
        
        # Origin
        scout_points.append(np.zeros(dim))
        
        # Standard candidates
        scout_points.append(np.ones(dim))
        scout_points.append(-np.ones(dim))
        
        # Corners
        for i in range(2**dim):
            p = np.zeros(dim)
            for j in range(dim):
                if (i >> j) & 1:
                    p[j] = 5.0
                else:
                    p[j] = -5.0
            scout_points.append(p)
            
        # Mid-points
        for i in range(2**dim):
            p = np.zeros(dim)
            for j in range(dim):
                if (i >> j) & 1:
                    p[j] = 2.5
                else:
                    p[j] = -2.5
            scout_points.append(p)

        # Evaluate all scout points
        # We have 500 budget. 1 + 2 + 32 + 32 = 67 points.
        # Leaves 433 for local optimization.
        
        for p in scout_points:
            if time.time() > deadline or eval_count >= budget:
                break
            try:
                fp = wrapped_func(p)
            except Exception:
                break

        # Phase 2: Local Optimization with Powell
        # Start Powell from the best point found in Phase 1.
        # Powell's method is derivative-free and good for Rosenbrock.
        
        if best_x is not None:
            x0 = best_x.copy()
            
            # We can try a few local runs from the top 3 scout points if budget allows?
            # Or just one strong run.
            # Powell in 5D is efficient.
            
            stop_flag = {'stop': False}
            
            def callback(xk):
                nonlocal eval_count
                if time.time() > deadline or eval_count >= budget:
                    stop_flag['stop'] = True
                    return True
                return False

            try:
                res = minimize(
                    wrapped_func, 
                    x0, 
                    method='Powell', 
                    options={
                        'xtol': 1e-8, 
                        'ftol': 1e-10,
                        'maxiter': 1000,
                        'maxfev': budget - eval_count - 10 # Reserve some budget
                    },
                    callback=callback
                )
                
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                pass
        
        # If we still have budget and time, maybe do a second Powell run from the 2nd best scout point?
        # For now, let's stick to one strong run to ensure we finish safely.
        
        if best_x is None:
             # Fallback: should not happen
             best_x = np.zeros(dim)
             best_f = func(best_x)
             eval_count += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
