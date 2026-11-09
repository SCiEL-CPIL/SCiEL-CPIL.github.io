"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, tuned for Bent Cigar anisotropy."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        calls_used = 0
        best_val = np.inf
        best_x = np.zeros(dim)

        def call_func(x):
            nonlocal calls_used, best_val, best_x
            if calls_used >= budget:
                return best_val
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            val = float(func(x))
            calls_used += 1
            if val < best_val:
                best_val = val
                best_x = x.copy()
            return val

        rng = np.random.default_rng(42)
        
        # Start at center, which is a reasonable guess for shifted instances
        x = np.zeros(dim)
        fx = call_func(x)
        
        sigma = 1.0
        
        for _ in range(budget - 1):
            if calls_used >= budget or time.time() > deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = call_func(y)
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.3
            else:
                sigma *= 0.85
                
            # Clamp sigma to avoid explosion or vanishing
            sigma = np.clip(sigma, 1e-8, 10.0)

        return best_x, best_val
# EVOLVE-BLOCK-END
