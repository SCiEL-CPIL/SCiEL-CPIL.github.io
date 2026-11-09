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
        time_limit = 300.0
        safety_margin = 5.0
        
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        sigma = 2.0
        for _ in range(self.budget - 1):
            if time.time() - start_time > time_limit - safety_margin:
                break
            z = rng.standard_normal(self.dim)
            x_new = np.clip(x + sigma * z, -5, 5)
            fx_new = func(x_new)
            if fx_new < fx:
                x = x_new
                fx = fx_new
                sigma *= 1.1
            else:
                sigma /= 1.1
        return x, fx
# EVOLVE-BLOCK-END
