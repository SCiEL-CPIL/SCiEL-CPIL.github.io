"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive step size for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        def remaining():
            return self.budget - calls_used
        
        def time_ok():
            return time.time() < deadline
        
        # Multi-start strategy: try several random initializations
        # For Büche-Rastrigin, we want to explore different basins
        n_starts = 5
        
        for start_idx in range(n_starts):
            if not time_ok() or remaining() <= 0:
                break
            
            # Generate starting point - spread across the domain
            if start_idx == 0:
                x = rng.uniform(-4, 4, self.dim)
            else:
                # Use quasi-random or stratified points for diversity
                x = rng.uniform(-4.5, 4.5, self.dim)
            
            fx = func(x)
            calls_used += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            # Run (1+1)-ES from this starting point
            sigma = 1.5
            sigma_min = 0.01
            sigma_max = 3.0
            
            while time_ok() and remaining() > 0:
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_used += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.25, sigma_max)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma = max(sigma * 0.85, sigma_min)
                
                # If step size is very small, consider restarting
                if sigma <= sigma_min * 1.1:
                    break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
