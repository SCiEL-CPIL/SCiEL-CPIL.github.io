"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step-size using 1/5 success rule."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit, 20s safety margin
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Start at a random point in the box
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        sigma = 3.0  # Initial step size - larger to explore quickly
        successes = 0
        attempts = 0

        for i in range(budget - 1):
            # Check time limit
            if time.time() > deadline - 5.0:
                break

            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)

            if fy < fx:
                x, fx = y, fy
                successes += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                pass

            attempts += 1

            # 1/5 success rule: adjust sigma based on success rate
            if attempts >= 5:
                success_rate = successes / attempts
                if success_rate > 0.2:
                    sigma *= 1.5
                elif success_rate < 0.2:
                    sigma *= 0.7
                successes = 0
                attempts = 0

            # Keep sigma in reasonable bounds
            sigma = max(0.001, min(sigma, 10.0))

        return best_x, best_f
# EVOLVE-BLOCK-END
