import numpy as np
import time
import scipy.optimize

class Optimizer:
    """Multi-start L-BFGS-B optimizer for BBOB functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Safety margin: 330s limit, stop at 325s
        deadline = time.time() + 325
        
        best_x = None
        best_f = np.inf
        calls = 0
        
        bounds = [(-5, 5) for _ in range(dim)]
        
        # Wrapper to track calls and update best
        def wrapped_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return np.inf
            val = func(x)
            calls += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Strategy: Multi-start with L-BFGS-B
        # We use 3-5 restarts to balance exploration and exploitation.
        # L-BFGS-B is efficient for smooth functions (Sphere, Rosenbrock) 
        # and reasonable for smooth multimodal (Rastrigin, Gallagher) if started in a basin.
        
        n_restarts = 4
        
        for i in range(n_restarts):
            if calls >= budget or time.time() > deadline:
                break
            
            # Generate random start
            x0 = rng.uniform(-5, 5, dim)
            
            # Evaluate start
            fx0 = wrapped_func(x0)
            
            if calls >= budget or time.time() > deadline:
                break
            
            # Determine budget for this local optimization
            # Leave some budget for subsequent restarts
            remaining_restarts = n_restarts - i
            # Heuristic: allocate remaining budget evenly
            local_budget = max(20, (budget - calls) // remaining_restarts)
            
            # Run L-BFGS-B
            # maxfev limits the number of function evaluations
            # maxiter limits the number of iterations
            try:
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={
                        'maxiter': 30, 
                        'maxfev': local_budget, 
                        'ftol': 1e-12, 
                        'gtol': 1e-8, 
                        'disp': False
                    }
                )
            except Exception:
                pass

        # Final polish if budget remains
        if calls < budget and best_x is not None and time.time() < deadline:
            remaining = budget - calls
            try:
                res = scipy.optimize.minimize(
                    wrapped_func, 
                    best_x, 
                    method='L-BFGS-B', 
                    bounds=bounds,
                    options={
                        'maxiter': 50, 
                        'maxfev': remaining, 
                        'ftol': 1e-14, 
                        'gtol': 1e-10, 
                        'disp': False
                    }
                )
            except Exception:
                pass

        return best_x, best_f
