import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        t_start = time.time()
        max_time = 290.0

        call_count = 0
        best_x = np.zeros(n)
        best_f = np.inf
        
        def wrapped_func(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return np.inf
            if time.time() - t_start > max_time:
                return np.inf
            
            call_count += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # Try multiple starting points to handle potential issues, though f6 is unimodal
        # But the shift/rotation might make it look non-convex if we start far away? No, sphere is convex.
        # However, Powell can get stuck in local minima if the function is not strictly convex? 
        # f6 is a sphere, so it is strictly convex.
        
        # Initialize with center
        x0 = np.zeros(n)
        
        # Run Powell's method
        # maxiter is not strictly calls, so we rely on the wrapper to stop.
        # However, Powell might take many iterations.
        # Let's try Nelder-Mead first, it's robust.
        # Or use a custom DE.
        
        # Let's stick to a robust black-box optimizer from scipy.
        # Powell is often better for low dim.
        res = minimize(wrapped_func, x0, method='Powell', options={'maxiter': budget})
        
        return best_x, best_f
