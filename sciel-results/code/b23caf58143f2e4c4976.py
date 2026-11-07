"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with final polish phase."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 300.0
        
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        calls = 1
        
        sigma = 2.0
        main_iterations = self.budget - 10
        
        for _ in range(main_iterations):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
            else:
                sigma *= 0.87
        
        remaining_calls = self.budget - calls
        if remaining_calls > 0:
            sigma_polish = 1e-3
            for _ in range(remaining_calls):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma_polish * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
        return x, fx
# EVOLVE-BLOCK-END
