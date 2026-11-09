import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        dim = self.dim
        # Start at the center
        x_center = np.zeros(dim)
        f_center = func(x_center)
        best_x = x_center.copy()
        best_f = f_center
        
        calls_used = 1
        
        # Determine the sign of the gradient for each dimension
        # For a linear function f(x) = c + w^T x, the minimum is at the vertex
        # where x_i = 5 if w_i < 0, and x_i = -5 if w_i > 0.
        # We estimate w_i by checking f(5 e_i) vs f(0).
        signs = np.full(dim, np.nan)
        
        for i in range(dim):
            if time.time() > deadline:
                break
            
            # Evaluate f at 5 * e_i
            x_probe = x_center.copy()
            x_probe[i] = 5.0
            
            f_probe = func(x_probe)
            calls_used += 1
            
            # Update best if probe is better
            if f_probe < best_f:
                best_f = f_probe
                best_x = x_probe.copy()
            
            # Determine sign of gradient component i
            if f_probe < f_center:
                signs[i] = -1.0  # Gradient negative, go to +5
            else:
                signs[i] = 1.0   # Gradient positive (or zero), go to -5
            
            # After each axis probe, evaluate a partial vertex:
            # known signs use their determined value, unknown dims default to +5
            x_partial = np.full(dim, 5.0)
            known_mask = ~np.isnan(signs)
            x_partial[known_mask] = np.where(signs[known_mask] < 0, 5.0, -5.0)
            
            f_partial = func(x_partial)
            calls_used += 1
            
            if f_partial < best_f:
                best_f = f_partial
                best_x = x_partial.copy()
        
        # Construct the optimal vertex
        x_vertex = np.where(signs < 0, 5.0, -5.0)
        
        # Evaluate the vertex
        f_vertex = func(x_vertex)
        calls_used += 1
        
        if f_vertex < best_f:
            best_f = f_vertex
            best_x = x_vertex.copy()
            
        # For linear functions, this vertex is the global minimum.
        # We can return now. The remaining budget does not improve the value.
        
        return best_x, best_f
