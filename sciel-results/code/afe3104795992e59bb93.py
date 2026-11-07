"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive sigma and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 280.0  # 50s safety margin before 330s kill

        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        sigma = 2.0
        calls = 1

        while calls < self.budget:
            if time.time() >= deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.3
            else:
                sigma *= 0.9

        return x, fx
# EVOLVE-BLOCK-END
