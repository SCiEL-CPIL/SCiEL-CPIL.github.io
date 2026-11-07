"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step sizing and final polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 330s kill
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        sigma = 1.5
        min_sigma = 1e-7
        max_sigma = 4.0
        budget_used = 1

        # Phase 1: main (1+1)-ES with adaptive sigma
        while budget_used < self.budget - 50:
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            budget_used += 1
            if fy < fx:
                x, fx = y, fy
                sigma = min(sigma * 1.2, max_sigma)
            else:
                sigma = max(sigma * 0.85, min_sigma)

        # Phase 2: final polish with very small steps
        if budget_used < self.budget and time.time() < deadline:
            sigma_polish = max(sigma * 0.1, 1e-6)
            for _ in range(self.budget - budget_used):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma_polish * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                budget_used += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma_polish = min(sigma_polish * 1.1, sigma * 0.5)
                else:
                    sigma_polish = max(sigma_polish * 0.9, 1e-8)

        return x, fx
# EVOLVE-BLOCK-END
