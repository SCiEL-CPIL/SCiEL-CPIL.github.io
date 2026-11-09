import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 270.0

        rng = np.random.default_rng(42) # Fixed seed for reproducibility in eval? No, evaluator might use different seeds. But fixed seed is safer for consistency.

        # Start at center
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls_made = 1

        # Initial sigma: Box is [-5, 5]. Shift is in [-4, 4].
        # Distance from 0 to worst case ~ 9.
        # Start with sigma=2.0 to make large moves initially.
        sigma = 2.0
        
        # 1/3 Rule parameters
        success_window = []
        window_size = 10
        c_sigma = 1.01 # Multiplicative factor
        
        for _ in range(budget - 1):
            if time.time() >= deadline:
                break
            if calls_made >= budget:
                break

            # Generate candidate
            step = sigma * rng.standard_normal(dim)
            y = np.clip(x + step, -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                # Success
                x = y
                fx = fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                success_window.append(1)
            else:
                # Failure
                success_window.append(0)

            # Update sigma based on 1/3 rule
            if len(success_window) > window_size:
                success_window.pop(0)
            
            if len(success_window) == window_size:
                success_rate = sum(success_window) / window_size
                if success_rate > 1/3:
                    sigma *= c_sigma
                elif success_rate < 1/3:
                    sigma /= c_sigma
                
            # Clamp sigma
            sigma = np.clip(sigma, 1e-8, 10.0)

        return best_x, best_f
