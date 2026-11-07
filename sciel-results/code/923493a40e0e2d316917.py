"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, deadline guard, and adaptive restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        calls_made = 1
        sigma = 2.0

        # Track best
        x_best = x.copy()
        f_best = fx

        # Stagnation tracking
        stagnation_count = 0
        stagnation_threshold = 30

        while calls_made < budget:
            if time.time() > deadline:
                break

            # Generate neighbor
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                # Accept
                x = y
                fx = fy
                sigma *= 1.5
                stagnation_count = 0

                # Update global best
                if fy < f_best:
                    f_best = fy
                    x_best = y.copy()
            else:
                # Reject
                sigma *= 0.87
                stagnation_count += 1

            # Check for stagnation - restart from best with larger step
            if stagnation_count >= stagnation_threshold:
                # Reset to best point with a larger sigma for exploration
                x = x_best.copy()
                fx = f_best
                sigma = 3.0
                stagnation_count = 0

            # If sigma becomes too small, restart with larger step
            if sigma < 1e-8 and calls_made < budget:
                x = x_best.copy()
                fx = f_best
                sigma = 2.0
                stagnation_count = 0

        # Final high-precision polish pass if budget remains
        remaining = budget - calls_made
        if remaining > 10 and time.time() < deadline:
            x = x_best.copy()
            fx = f_best
            sigma_polish = 0.05

            for _ in range(min(remaining, 50)):
                if time.time() > deadline or calls_made >= budget:
                    break

                y = np.clip(x + sigma_polish * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x = y
                    fx = fy
                    sigma_polish *= 1.2
                    if fy < f_best:
                        f_best = fy
                        x_best = y.copy()
                else:
                    sigma_polish *= 0.85

        return x_best, f_best
# EVOLVE-BLOCK-END
