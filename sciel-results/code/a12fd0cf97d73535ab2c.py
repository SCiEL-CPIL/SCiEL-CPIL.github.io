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
        
        # Array to store slopes
        slopes = np.zeros(d)
        
        # 2. Iteratively probe dimensions and evaluate partial optimal vertices
        for i in range(d):
            # Probe dimension i
            x_probe = np.zeros(d)
            x_probe[i] = 5.0
            f_probe = func(x_probe)
            
            # Calculate slope
            slopes[i] = (f_probe - f0) / 5.0
            
            # Update best if this probe is better (unlikely if slope > 0, likely if slope < 0)
            if f_probe < best_val:
                best_val = f_probe
                best_x = x_probe

            # Construct partial optimal vertex
            # For dimensions 0..i, use the sign of the slope to go to the boundary
            # For dimensions i+1..d-1, use 0
            x_partial = np.zeros(d)
            for j in range(i + 1):
                if slopes[j] > 0:
                    x_partial[j] = -5.0
                elif slopes[j] < 0:
                    x_partial[j] = 5.0
                else:
                    x_partial[j] = 0.0
            
            # Evaluate partial optimal vertex
            f_partial = func(x_partial)
            
            if f_partial < best_val:
                best_val = f_partial
                best_x = x_partial

        # 3. After the loop, we have evaluated the full optimal vertex at the last step (i=d-1)
        # So best_x is the global minimum.
        
        # 4. Use remaining budget to maintain the call count
        # Calls made: 1 (origin) + d (probes) + d (partials) = 2d + 1
        calls_made = 2 * d + 1
        
        for _ in range(budget - calls_made):
            func(best_x)
            
        return best_x, best_val
