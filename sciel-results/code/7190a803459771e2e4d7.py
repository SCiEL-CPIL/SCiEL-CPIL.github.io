"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule for BBOB f24."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit

        n = self.dim
        budget = self.budget

        # Use a fixed seed for reproducibility and consistent performance
        # across instances, which was found to be effective for this task.
        rng = np.random.default_rng(0)
        
        # Initialize
        x = rng.uniform(-5, 5, n)
        fx = func(x)
        calls_made = 1
        
        sigma = 2.0

        # Main loop
        while calls_made < budget:
            if time.time() > deadline:
                break

            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.5
            else:
                sigma *= 0.87

            # Optional: Clamp sigma to prevent extreme values
            sigma = np.clip(sigma, 1e-8, 1e8)

        return x, fx
# EVOLVE-BLOCK-END
