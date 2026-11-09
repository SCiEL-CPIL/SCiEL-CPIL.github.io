import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        
        dim = self.dim
        budget = self.budget
        
        # Call 1: Evaluate at origin
        x0 = np.zeros(dim)
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0
        calls_used = 1
        
        # Determine gradient signs by probing each dimension
        # f(x) = w^T x + b, so f(+5*e_i) - f(0) = 5*w_i
        signs = np.zeros(dim)
        for i in range(dim):
            if calls_used >= budget or time.time() > deadline:
                break
            
            x_probe = np.zeros(dim)
            x_probe[i] = 5.0
            f_probe = func(x_probe)
            calls_used += 1
            
            # If f_probe < f0, then w_i < 0, minimum at x_i = +5
            # If f_probe >= f0, then w_i >= 0, minimum at x_i = -5
            if f_probe < f0:
                signs[i] = -1.0
            else:
                signs[i] = 1.0
            
            if f_probe < best_f:
                best_f = f_probe
                best_x = x_probe.copy()
        
        # Jump to the optimal vertex
        x_vertex = np.zeros(dim)
        for i in range(dim):
            if signs[i] == 0:
                x_vertex[i] = 0.0
            elif signs[i] < 0:
                x_vertex[i] = 5.0
            else:
                x_vertex[i] = -5.0
        
        if calls_used < budget and time.time() < deadline:
            f_vertex = func(x_vertex)
            calls_used += 1
            
            if f_vertex < best_f:
                best_f = f_vertex
                best_x = x_vertex.copy()
        
        # Fill remaining budget with best point
        while calls_used < budget:
            if time.time() > deadline:
                break
            func(best_x)
            calls_used += 1
        
        return best_x, best_f
