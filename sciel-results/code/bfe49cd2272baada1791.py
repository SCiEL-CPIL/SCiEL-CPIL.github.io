import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit
        
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Start at origin - expected center of shift distribution
        x = np.zeros(d)
        fx = func(x)
        count = 1
        sigma = 2.5
        
        # Main (1+1)-ES phase with gentle adaptation
        # Smaller factors for more stable, gradual convergence
        while count < budget:
            if time.time() > deadline:
                break
                
            y = x + sigma * rng.standard_normal(d)
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            count += 1
            
            if fy < fx:
                x = y
                fx = fy
                sigma *= 1.1
            else:
                sigma *= 0.95
            
            # Transition to polish phase when sigma is small
            if sigma < 1e-4:
                break
        
        # Polish phase: use remaining budget for high-precision local search
        # with very small steps to squeeze out final improvements
        if count < budget and time.time() < deadline:
            polish_sigma = max(sigma, 1e-6)
            while count < budget:
                if time.time() > deadline:
                    break
                    
                y = x + polish_sigma * rng.standard_normal(d)
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                count += 1
                
                if fy < fx:
                    x = y
                    fx = fy
                    polish_sigma *= 1.05
                else:
                    polish_sigma *= 0.98
                
                if polish_sigma < 1e-12:
                    break
        
        return x, fx
