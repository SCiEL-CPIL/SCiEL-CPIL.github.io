"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-Start Nelder-Mead Optimizer."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 300s limit, 10s safety margin
        
        rng = np.random.default_rng(42)
        dim = self.dim
        
        # We will use a portion of the budget for global exploration (initial points)
        # and the rest for local refinement.
        # With budget=500, let's try ~20-50 initial points.
        # Nelder-Mead in 5D is fast. 
        # Let's allocate 30 initial points.
        n_starts = 30
        
        best_x = None
        best_f = np.inf
        
        # Generate initial points
        # Mix of random uniform and a few structured points (center, corners)
        inits = []
        
        # Center
        inits.append(np.zeros(dim))
        
        # Random points
        for _ in range(n_starts - 1):
            inits.append(rng.uniform(-5, 5, dim))
        
        # Shuffle to ensure diversity in ordering if time runs out
        rng.shuffle(inits)
        
        calls_used = 0
        max_calls_per_opt = 150 # Roughly 500 / 30
        
        for i, x0 in enumerate(inits):
            if time.time() > deadline - 5.0:
                break
            
            # Check budget
            if calls_used >= self.budget - 10: # Leave some buffer
                break
                
            # Define a wrapper to count calls and enforce time
            local_calls = 0
            def wrapped_func(x):
                nonlocal local_calls, calls_used
                if time.time() > deadline:
                    raise ValueError("Time limit exceeded")
                local_calls += 1
                calls_used += 1
                if local_calls > max_calls_per_opt:
                    # Return a very bad value to stop early? 
                    # Or just let it run until maxiter. 
                    # scipy minimize will stop if maxiter is reached.
                    pass
                return func(x)
            
            try:
                res = minimize(wrapped_func, x0, method='Nelder-Mead', 
                               options={'maxiter': max_calls_per_opt, 'xatol': 1e-4, 'fatol': 1e-4})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x
            except Exception:
                # If an exception occurs, we keep the current best
                pass

        # If no optimization ran or failed, return a random point? 
        # But we must have made at least one call.
        # The loop above ensures calls are made.
        
        return best_x, best_f
# EVOLVE-BLOCK-END
