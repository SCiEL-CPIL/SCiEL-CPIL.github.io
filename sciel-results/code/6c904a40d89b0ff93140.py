import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        deadline = time.time() + 290
        
        # 1. Evaluate center
        x_center = np.zeros(dim)
        f_center = func(x_center)
        f_best = f_center
        x_best = x_center.copy()
        
        # 2. Probe each axis from center to determine gradient signs
        # f(5*e_i) = 5*g_i + c
        # g_i = (f(5*e_i) - f(0)) / 5
        gradients = np.zeros(dim)
        for i in range(dim):
            if time.time() > deadline:
                break
            
            x_probe = np.zeros(dim)
            x_probe[i] = 5.0
            f_probe = func(x_probe)
            
            # Update best so far
            if f_probe < f_best:
                f_best = f_probe
                x_best = x_probe.copy()
            
            # Calculate gradient component
            gradients[i] = (f_probe - f_center) / 5.0
            
        # 3. Construct optimal vertex
        # If g_i > 0, x_i should be -5
        # If g_i < 0, x_i should be 5
        # If g_i == 0, either is fine (use 0 or 5, doesn't matter for linear)
        x_opt = np.zeros(dim)
        for i in range(dim):
            if gradients[i] > 0:
                x_opt[i] = -5.0
            else:
                x_opt[i] = 5.0
        
        # 4. Evaluate optimal vertex
        if time.time() < deadline:
            f_opt = func(x_opt)
            if f_opt < f_best:
                f_best = f_opt
                x_best = x_opt.copy()
        
        return x_best, f_best
