"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive (1+1)-ES with deadline guard and final polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initialize at center (the attractive sector function f6 typically
        # has its minimum near the origin after shift/rotation, and center
        # is a safe robust starting point)
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        sigma = 3.0  # initial step size covering a good fraction of the box

        calls = 1
        # 1/5 success rule parameters
        inc_factor = 1.5
        dec_factor = 0.87
        # Track successes for adaptive sigma
        successes = 0
        window = 10

        for i in range(budget - 1):
            # Deadline guard
            if time.time() > deadline:
                break

            # Generate candidate
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x = y
                fx = fy
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                successes += 1
            else:
                pass

            # Adaptive step size using 1/5 rule on a sliding window
            # Simple approach: adjust every few steps
            if (i + 1) % window == 0:
                if successes >= window * 0.2:
                    sigma *= inc_factor
                else:
                    sigma *= dec_factor
                successes = 0

            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 1e-6, 10.0)

        # Final polish: if we have remaining budget and time, do a fine local search
        remaining_time = deadline - time.time()
        if remaining_time > 5.0 and calls < budget:
            # Do a small-scale refinement around the best point
            sigma_polish = max(sigma * 0.1, 1e-4)
            for j in range(min(50, budget - calls)):
                if time.time() > deadline:
                    break
                y = best_x + sigma_polish * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                if fy < best_fx:
                    best_x = y
                    best_fx = fy
                    sigma_polish *= 0.8
                else:
                    sigma_polish *= 0.9
                sigma_polish = max(sigma_polish, 1e-8)

        return best_x, best_fx
# EVOLVE-BLOCK-END
