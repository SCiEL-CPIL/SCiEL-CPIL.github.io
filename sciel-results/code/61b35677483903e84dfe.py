import numpy as np
from scipy.optimize import differential_evolution
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
            start_time = time.time()
            deadline = start_time + 280
            
            n = self.dim
            bounds = [(-5, 5)] * n
            
            # We need to track calls to ensure we don't exceed budget
            # differential_evolution doesn't expose a callback for every single function call easily
            # in older scipy versions, but newer ones might.
            # However, we can wrap the function.
            
            calls = [0]
            best_val = [np.inf]
            
            def wrapped_func(x):
                if time.time() > deadline:
                    raise Exception("Time limit")
                calls[0] += 1
                val = func(x)
                if val < best_val[0]:
                    best_val[0] = val
                return val
            
            # Settings
            # Popsize 5 -> 25 individuals.
            # 500 calls / 25 = 20 iterations.
            # maxiter should be high enough, but we limit by calls.
            # We can use a callback to stop if we exceed budget.
            
            def callback(xk, convergence):
                return calls[0] >= self.budget - 10 # Leave some margin
            
            try:
                res = differential_evolution(
                    wrapped_func, 
                    bounds, 
                    maxiter=1000, 
                    popsize=5, 
                    tol=1e-8, 
                    init='latinhypercube', 
                    mutation=(0.5, 1.0), 
                    recombination=0.5, 
                    seed=42,
                    callback=callback,
                    polish=True # Nelder-Mead polish at the end? 
                )
            except Exception as e:
                # If time limit or other error, return best found
                pass
            
            # differential_evolution returns the best x and f
            # But we need to return the best x and f found during the process?
            # The prompt says "return value of __call__ is ignored".
            # So we just need to make the calls.
            
            # Wait, the prompt says "return best_x, best_fx".
            # Even though it's ignored, we should return something sensible.
            
            if res:
                return res.x, res.fun
            else:
                # Fallback
                return np.zeros(n), best_val[0]
