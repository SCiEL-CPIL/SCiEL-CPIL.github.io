# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive sigma, periodic global exploration, and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s limit

        rng = np.random.default_rng(0)

        # Global best
        x_best = rng.uniform(-5, 5, self.dim)
        fx_best = func(x_best)

        calls_made = 1
        remaining = self.budget - 1

        # Parameters
        sigma = 1.0
        sigma_min = 1e-4
        sigma_max = 15.0
        stall_limit = 25
        stall_count = 0
        exploration_interval = 40  # Periodically explore new region

        while remaining > 0 and time.time() < deadline:
            # Check if we should do a global exploration step
            if stall_count > 0 and (stall_count >= stall_limit or calls_made % exploration_interval == 0):
                # Try a new random point
                x_new = rng.uniform(-5, 5, self.dim)
                fx_new = func(x_new)
                calls_made += 1
                remaining -= 1

                if fx_new < fx_best:
                    x_best, fx_best = x_new, fx_new
                    sigma = max(sigma * 0.5, sigma_min)
                    stall_count = 0
                else:
                    # If exploration didn't help, reset sigma and continue
                    sigma = 1.0
                    stall_count = 0
                continue

            # Normal (1+1)-ES step
            y = np.clip(x_best + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls_made += 1
            remaining -= 1

            if fy < fx_best:
                x_best, fx_best = y, fy
                sigma = min(sigma * 1.15, sigma_max)
                stall_count = 0
            else:
                sigma = max(sigma * 0.85, sigma_min)
                stall_count += 1

        return x_best, fx_best
# EVOLVE-BLOCK-END
