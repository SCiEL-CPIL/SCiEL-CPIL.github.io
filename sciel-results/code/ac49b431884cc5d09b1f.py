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
        deadline = start_time + 290.0  # 30s safety margin before 320s limit
        rng = np.random.default_rng(0)
        dim = self.dim

        # Start at origin (reasonable prior since minimizer is in [-4,4]^5)
        x = np.zeros(dim)
        fx = func(x)
        calls_made = 1

        sigma = 2.0
        # Success-rate based sigma adaptation
        n_success = 0
        n_total = 0
        window_size = 8

        for i in range(self.budget - calls_made):
            if time.time() >= deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                x, fx = y, fy
                n_success += 1
            n_total += 1

            # Adapt sigma based on success rate over the window
            if n_total >= window_size:
                success_rate = n_success / n_total
                # Target success rate around 1/5 = 0.2
                if success_rate > 0.25:
                    sigma *= 1.15
                elif success_rate < 0.15:
                    sigma *= 0.85
                # else: keep same
                n_success = 0
                n_total = 0

            # Keep sigma in reasonable bounds
            sigma = np.clip(sigma, 0.005, 5.0)

        return x, fx
# EVOLVE-BLOCK-END
