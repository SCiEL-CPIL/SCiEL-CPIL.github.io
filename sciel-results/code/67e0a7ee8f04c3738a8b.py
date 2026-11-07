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
        deadline = start_time + 280.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Adaptive step size
        sigma = 1.5
        sigma_min = 1e-6
        sigma_max = 4.0

        # Success tracking for 1/5 rule
        successes = 0
        total = 0
        window_size = 15

        calls_made = 1

        for i in range(budget - 1):
            # Check deadline
            if time.time() >= deadline:
                break

            # Generate trial point
            direction = rng.standard_normal(dim)
            y = np.clip(x + sigma * direction, -5, 5)
            fy = func(y)
            calls_made += 1

            # Update best
            if fy < best_fx:
                best_x = y.copy()
                best_fx = fy

            # 1/5 rule for step size adaptation
            if fy < fx:
                successes += 1
                x = y
                fx = fy
            else:
                # Try a few more directions before accepting
                # Actually, for (1+1) ES we just reject
                pass

            total += 1

            # Adapt sigma every window_size steps
            if total % window_size == 0:
                rate = successes / total
                if rate > 1.0 / 5.0:
                    sigma *= 1.3
                elif rate < 1.0 / 5.0:
                    sigma *= 0.8
                sigma = np.clip(sigma, sigma_min, sigma_max)
                successes = 0
                total = 0

        return best_x, best_fx
# EVOLVE-BLOCK-END
