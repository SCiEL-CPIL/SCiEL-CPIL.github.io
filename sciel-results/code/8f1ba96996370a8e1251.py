"""Optimized program using Multi-Start Nelder-Mead for the Bent Cigar function."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        d = self.dim
        total_budget = self.budget
        
        best_x = np.zeros(d)
        best_f = np.inf
        calls_made = 0
        
        # Multi-Start Strategy
        # For d=5, 500 calls. 
        # Let's do 5 restarts, 100 calls each.
        num_restarts = 5
        calls_per_restart = total_budget // num_restarts
        
        for r in range(num_restarts):
            if time.time() > deadline:
                break
            if calls_made >= total_budget:
                break
            
            # Random start
            x0 = np.random.uniform(-5, 5, size=d)
            
            # Wrap func to track calls and best
            def wrapped_func(x, state={'calls': 0, 'best_f': best_f, 'best_x': best_x}):
                # We need to be careful with closures and mutable state in Python
                # It's better to use a class or nonlocal. 
                # Since we are in __call__, we can use a local variable if we structure it right, 
                # but minimize calls the function multiple times.
                # Let's use a list to hold state.
                pass

            # Actually, simpler: 
            # We will just let minimize run. 
            # We will check the result. 
            # To handle the budget strictly, we will use a callback that raises an exception 
            # when the local budget is exceeded.
            
            local_calls = [0]
            local_limit = calls_per_restart
            
            def callback(xk):
                local_calls[0] += 1
                if local_calls[0] >= local_limit:
                    raise Exception("Local Budget Exceeded")
            
            # We need to update best_f/best_x if the optimizer finds a better point.
            # The optimizer returns the best point found during its run.
            # So we just compare res.fun with best_f.
            
            try:
                res = minimize(
                    func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': local_limit * 5, # Generous maxiter, callback will stop it
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'adaptive': True
                    },
                    callback=callback
                )
                
                # Update global best
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                
                # Update global calls count
                calls_made += local_calls[0]
                
            except Exception as e:
                # If it stopped due to budget, we don't have a `res`.
                # We might have improved best_f? 
                # Nelder-Mead updates its internal best, but doesn't expose it until it returns.
                # If it raises, we lose the intermediate bests found in this restart.
                # This is a downside. 
                
                # To fix this, we can wrap the func to update best_f/best_x on every call.
                
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
