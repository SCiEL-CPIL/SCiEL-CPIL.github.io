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
        deadline = start_time + 290  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        sigma = 2.0
        calls = 1

        # (1+1)-ES with adaptive step size
        for _ in range(self.budget - 1):
            # Check time deadline
            if time.time() >= deadline:
                break

            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            # Update
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
                # Update best
                best_x = x.copy()
                best_fx = fx
            else:
                sigma *= 0.87

            # Clamp sigma to reasonable range
            sigma = max(1e-8, min(sigma, 10.0))

        # Final polish: local refinement around best point
        if time.time() < deadline - 5:
            # Try a few more steps with smaller sigma
            sigma_polish = max(1e-6, sigma * 0.1)
            for _ in range(20):
                if time.time() >= deadline:
                    break
                y = np.clip(best_x + sigma_polish * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < best_fx:
                    best_x = y.copy()
                    best_fx = fy
                    sigma_polish *= 1.2
                else:
                    sigma_polish *= 0.7

        return best_x, best_fx
# EVOLVE-BLOCK-END
