"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive (1+1)-ES with time guard for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        sigma = 2.0

        # Adaptive step size with success rate tracking
        successes = 0
        total_steps = 0
        window_size = 15

        for i in range(budget - 1):
            # Check time limit
            if time.time() > deadline:
                break

            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            total_steps += 1

            if fy < fx:
                x, fx = y, fy
                successes += 1
                # Increase step size on success
                sigma *= 1.3
                if fx < best_fx:
                    best_x = x.copy()
                    best_fx = fx
            else:
                # Decrease step size on failure
                sigma *= 0.85

            # Adapt sigma based on success rate over recent window
            if total_steps % window_size == 0:
                success_rate = successes / window_size
                # Target ~30% success rate for optimal progress
                if success_rate > 0.4:
                    sigma *= 1.2
                elif success_rate < 0.15:
                    sigma *= 0.8
                successes = 0

            # Ensure sigma stays in reasonable bounds
            sigma = np.clip(sigma, 1e-6, 10.0)

        return best_x, best_fx
# EVOLVE-BLOCK-END
