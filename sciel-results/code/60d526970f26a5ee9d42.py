"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, deadline guard, and final polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        
        # Initial point
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        sigma = 2.0
        calls = 1
        
        # Main search loop
        while calls < self.budget:
            if time.time() > deadline:
                break
            
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1
            
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
                if fy < best_fx:
                    best_x, best_fx = y.copy(), fy
            else:
                sigma *= 0.87
        
        # Final polish pass: try small steps around the best point
        # Use remaining budget for high-precision local search
        if calls < self.budget and time.time() < deadline - 5:
            # Reduce sigma for polish
            sigma_polish = sigma * 0.1
            x_polish = best_x.copy()
            fx_polish = best_fx
            for _ in range(min(50, self.budget - calls)):
                if time.time() > deadline:
                    break
                y = np.clip(x_polish + sigma_polish * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx_polish:
                    x_polish, fx_polish = y, fy
                    sigma_polish *= 1.2
                    if fy < best_fx:
                        best_x, best_fx = y.copy(), fy
                else:
                    sigma_polish *= 0.9
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
