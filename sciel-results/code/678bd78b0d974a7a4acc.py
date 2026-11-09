"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        sigma = 1.5
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        calls = 1
        best_x = x.copy()
        best_f = fx
        for _ in range(self.budget - 1):
            if time.time() >= deadline or calls >= self.budget:
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.25
            else:
                sigma *= 0.95
            if fy < best_f:
                best_f = fy
                best_x = y.copy()
        return best_x, best_f
# EVOLVE-BLOCK-END
