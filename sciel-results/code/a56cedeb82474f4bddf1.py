"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Start at origin (common for BBOB-style functions)
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        sigma = 1.0
        calls_made = 1

        # Adaptive step size with success counter
        # Use a simple 1/5 success rule variant
        successes = 0
        window = 10

        for i in range(budget - 1):
            # Deadline check
            if time.time() > deadline:
                break

            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                # Success
                x, fx = y, fy
                if fy < best_f:
                    best_x, best_f = y.copy(), fy
                successes += 1
                sigma *= 1.25
            else:
                # Failure
                sigma *= 0.85

            # Periodically adjust sigma based on success rate
            if (i + 1) % window == 0:
                success_rate = successes / window
                if success_rate > 0.5:
                    sigma *= 1.1
                elif success_rate < 0.2:
                    sigma *= 0.9
                successes = 0

            # Ensure sigma stays in reasonable bounds
            sigma = np.clip(sigma, 1e-6, 10.0)

        # Final polish: try a few very small steps around the best point
        if time.time() < deadline - 5:
            for _ in range(20):
                if time.time() > deadline:
                    break
                y = np.clip(best_x + 0.01 * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < best_f:
                    best_x, best_f = y.copy(), fy

        return best_x, best_f
# EVOLVE-BLOCK-END
