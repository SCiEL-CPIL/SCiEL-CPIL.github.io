"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, starting from center."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Deadline guard: 300s limit, stop at 290s to be safe
        start_time = time.time()
        deadline = start_time + 290.0
        
        rng = np.random.default_rng(0)
        # Start from center (0,0,...,0) as it is closer to the shifted optimum on average
        x = np.zeros(self.dim)
        fx = func(x)
        sigma = 2.0
        
        # 1/5 success rule parameters
        c_sigma = 1.0 / (np.sqrt(self.dim) * 2.0)
        c_sigma_1 = 1.0 + c_sigma * (np.sqrt(self.dim + 1.5) - 1)
        c_sigma_2 = 1.0 + c_sigma * (self.dim - 1)
        chi_d = np.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(32*self.dim**2))
        
        # Simple adaptive sigma
        # For (1+1)-ES, a common adaptation is:
        # if success: sigma *= 1.5
        # else: sigma *= 0.87
        # Or use the 1/5 rule logic explicitly.
        # Let's use a robust adaptive scheme.
        
        calls_used = 1
        for _ in range(self.budget - 1):
            if time.time() > deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_used += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
            else:
                sigma *= 0.87
                
            # Prevent sigma from becoming too small or too large
            sigma = np.clip(sigma, 1e-8, 10.0)
            
        return x, fx
# EVOLVE-BLOCK-END
