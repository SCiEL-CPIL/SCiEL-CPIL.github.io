import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Deadline guard: 300s limit, return before 330s wall clock kill
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin
        
        rng = np.random.default_rng(42)
        
        # Initial point
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        
        # Adaptive step size (1/5 success rule)
        sigma = 1.0
        successes = 0
        total_steps = 0
        
        # We use the full budget. 
        # (1+1)-ES is very fast per call (vectorized ops are trivial for dim=5).
        # 500 calls is small for time, but large enough for exploration.
        
        for i in range(self.budget - 1):
            # Time check
            if time.time() > deadline:
                break
            
            # Generate candidate
            y = x + sigma * rng.standard_normal(self.dim)
            # Clip to box [-5, 5]
            y = np.clip(y, -5, 5)
            
            fy = func(y)
            
            if fy < fx:
                x = y
                fx = fy
                successes += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            # else: x and fx remain unchanged
            
            total_steps += 1
            
            # Adaptive sigma adjustment every 10 steps
            if total_steps % 10 == 0:
                if successes / 10.0 > 0.2:
                    sigma *= 1.5
                elif successes / 10.0 < 0.2:
                    sigma /= 1.5
                # Clamp sigma to reasonable range
                sigma = np.clip(sigma, 0.01, 10.0)
                successes = 0
        
        return best_x, best_f
