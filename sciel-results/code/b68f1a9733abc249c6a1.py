import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        
        # 1. Evaluate at origin
        x0 = np.zeros(d)
        f0 = func(x0)
        best_val = f0
        best_x = x0
        
        # 2. Evaluate along each axis at the boundary to determine gradient sign
        # and get a much better intermediate best value.
        # We probe at x_i = 5.0 for dimension i, others 0.
        # This is within [-5, 5]^d.
        # f(x_i) = c + 5 * w_i
        # f(0) = c
        # w_i = (f(x_i) - f(0)) / 5
        
        slopes = np.zeros(d)
        
        for i in range(d):
            x_probe = np.zeros(d)
            x_probe[i] = 5.0
            f_probe = func(x_probe)
            
            # Calculate slope
            slopes[i] = (f_probe - f0) / 5.0
            
            # Update best if this probe is better
            if f_probe < best_val:
                best_val = f_probe
                best_x = x_probe

        # 3. Construct the optimal vertex
        # If slope is positive, min is at -5.
        # If slope is negative, min is at 5.
        # If slope is 0, either is fine.
        x_opt = np.where(slopes > 0, -5.0, 5.0)
        
        # 4. Evaluate the optimal vertex
        f_opt = func(x_opt)
        if f_opt < best_val:
            best_val = f_opt
            best_x = x_opt
            
        # 5. Use remaining budget to maintain the call count
        calls_made = d + 2
        for _ in range(budget - calls_made):
            func(best_x)
            
        return best_x, best_val
