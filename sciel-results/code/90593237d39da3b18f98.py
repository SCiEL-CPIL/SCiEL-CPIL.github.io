"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5th success rule for Büche-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 50s safety margin before 330s kill
        rng = np.random.default_rng(0)
        
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        sigma = 1.5
        budget_used = 1
        
        # 1/5th success rule: target success rate ~0.2
        # Adjust sigma by factor based on recent successes
        success_window = []
        window_size = 10
        
        while budget_used < self.budget:
            if time.time() > deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            budget_used += 1
            
            if fy < fx:
                x, fx = y, fy
                success_window.append(1)
            else:
                success_window.append(0)
            
            # Adapt sigma based on success rate in recent window
            if len(success_window) >= window_size:
                success_rate = np.mean(success_window[-window_size:])
                if success_rate > 0.2:
                    sigma *= 1.05  # too many successes, increase step
                elif success_rate < 0.1:
                    sigma *= 0.95  # too few successes, decrease step
                sigma = max(0.01, min(sigma, 10.0))
        
        return x, fx
# EVOLVE-BLOCK-END
