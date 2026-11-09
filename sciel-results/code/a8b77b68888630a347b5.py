"""Combined program: (1+1)-ES with Restarts + Nelder-Mead Refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.optimize


class Optimizer:
    """
    Hybrid optimizer:
    1. Global phase: (1+1)-ES with adaptive step size and restarts from best.
    2. Local phase: Nelder-Mead refinement on the best point found.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, use 280s to be safe
        deadline = start_time + 280
        
        n = self.dim
        calls = 0
        best_x = None
        best_f = np.inf
        
        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            if calls >= self.budget:
                raise StopIteration("Budget exceeded")
            
            x_arr = np.asarray(x)
            f_val = func(x_arr)
            calls += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_arr.copy()
            return f_val

        rng = np.random.default_rng(0)
        
        # Phase 1: Global Search with (1+1)-ES and Restarts
        # Allocate 60% of budget to global exploration to find a good basin
        global_budget = int(self.budget * 0.6)
        
        # Initial point
        x = rng.uniform(-5, 5, n)
        try:
            fx = safe_func(x)
        except (TimeoutError, StopIteration):
            return best_x, best_f
            
        sigma = 1.0
        stagnation = 0
        max_stagnation = 50
        
        for _ in range(global_budget):
            if calls >= self.budget or time.time() > deadline:
                break
                
            # Generate candidate
            y = x + sigma * rng.standard_normal(n)
            y = np.clip(y, -5, 5)
            
            try:
                fy = safe_func(y)
            except (TimeoutError, StopIteration):
                break
                
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.2
                stagnation = 0
            else:
                sigma *= 0.9
                stagnation += 1
                
                # If stagnating, restart from global best with larger step
                if stagnation > max_stagnation:
                    if best_x is not None:
                        x = best_x.copy()
                        fx = best_f
                    else:
                        x = rng.uniform(-5, 5, n)
                        try:
                            fx = safe_func(x)
                        except (TimeoutError, StopIteration):
                            break
                    sigma = 2.0
                    stagnation = 0
                    # Reduce threshold for more aggressive restarts later
                    max_stagnation = max(20, max_stagnation // 2)

        # Phase 2: Local Refinement with Nelder-Mead
        # Use remaining budget for local optimization
        if calls < self.budget and time.time() < deadline and best_x is not None:
            remaining_calls = self.budget - calls
            # Nelder-Mead uses ~dim+1 evaluations per iteration roughly
            # Be conservative with maxiter to stay within budget
            max_iter = max(5, remaining_calls // (n + 2))
            
            try:
                # Nelder-Mead is derivative-free and good for final polish
                result = scipy.optimize.minimize(
                    safe_func, 
                    best_x, 
                    method='Nelder-Mead', 
                    options={
                        'maxiter': max_iter,
                        'xatol': 1e-5,
                        'fatol': 1e-5,
                        'adaptive': True
                    }
                )
            except (TimeoutError, StopIteration, Exception):
                pass # Keep best from Phase 1 if this fails

        return best_x, best_f
# EVOLVE-BLOCK-END
