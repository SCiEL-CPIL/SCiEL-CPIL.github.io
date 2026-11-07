import numpy as np
import time
import scipy.optimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        
        n = self.dim
        budget = self.budget
        
        calls = 0
        best_f = np.inf
        best_x = None
        
        def wrapped_func(x):
            nonlocal calls
            if calls >= budget:
                return 1e10
            calls += 1
            x_clipped = np.clip(x, -5, 5)
            return func(x_clipped)
        
        # Use Differential Evolution for global search
        # This is more robust for multimodal functions like Gallagher 101 peaks
        bounds = [(-5, 5)] * n
        
        try:
            # popsize=10, maxiter=45 -> ~450 calls
            # Adjust if budget allows
            popsize = 10
            max_iter = 45
            
            res = scipy.optimize.differential_evolution(
                wrapped_func, 
                bounds, 
                maxiter=max_iter, 
                popsize=popsize, 
                tol=1e-5, 
                seed=42,
                polish=True, # Use Nelder-Mead to polish the best result
                init='sobol' 
            )
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x
        except Exception:
            pass

        # If we still have budget, do some local refinement with Nelder-Mead on the best found
        if calls < budget and time.time() < deadline - 5:
            max_iter = min(50, budget - calls)
            if max_iter > 0:
                try:
                    res = scipy.optimize.minimize(
                        wrapped_func, 
                        best_x, 
                        method='Nelder-Mead',
                        options={'maxiter': max_iter, 'xatol': 1e-5, 'fatol': 1e-5}
                    )
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x
                except Exception:
                    pass

        return best_x, best_f
