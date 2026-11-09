"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 290.0

        # Start from a random point in the box
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        calls = 1

        # (1+1)-ES with 1/5 success rule
        # sigma adaptation: increase on success, decrease on failure
        sigma = 2.0

        for _ in range(budget - 1):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
            else:
                sigma *= 0.87

        return x, fx
# EVOLVE-BLOCK-END
