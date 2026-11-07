"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with per-dimension step size adaptation and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        sigma = np.full(dim, 2.0)

        budget_used = 1
        for _ in range(budget - 1):
            if time.time() > deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            budget_used += 1

            if fy < fx:
                x, fx = y, fy
                # Increase sigma proportionally to progress in each dim
                for d in range(dim):
                    step = abs(y[d] - x[d])
                    if step > 1e-10:
                        sigma[d] = np.clip(sigma[d] * 1.15, 0.01, 10.0)
            else:
                # Decrease sigma for dimensions where we moved but didn't improve
                for d in range(dim):
                    if abs(y[d] - x[d]) > 1e-10:
                        sigma[d] = np.clip(sigma[d] * 0.88, 0.01, 10.0)

        return x, fx
# EVOLVE-BLOCK-END
