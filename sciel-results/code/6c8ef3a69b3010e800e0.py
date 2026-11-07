"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 300s limit

        rng = np.random.default_rng(0)
        # Initialize near the center, as BBOB shifts are usually moderate
        x = np.zeros(self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        sigma = 1.0
        
        # 1/5 rule parameters
        window_size = 10
        successes = []
        c_1_5 = 1.0  # Approximation for small dimensions
        
        for _ in range(self.budget - 1):
            if time.time() >= deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            
            # Track success for 1/5 rule
            if fy < fx:
                successes.append(1)
                x, fx = y, fy
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
            else:
                successes.append(0)
            
            # Maintain window size
            if len(successes) > window_size:
                successes.pop(0)
            
            # Update sigma
            if len(successes) == window_size:
                success_rate = np.mean(successes)
                sigma *= (1 + (1.0 / c_1_5) * (0.2 - success_rate))
            
            # Clamp sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-8, 10.0)

        return best_x, best_fx
# EVOLVE-BLOCK-END
