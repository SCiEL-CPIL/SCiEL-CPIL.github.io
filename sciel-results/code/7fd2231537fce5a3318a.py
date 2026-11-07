"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with zero-centered init, stable step-size adaptation, and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(0)
        x = np.zeros(self.dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        sigma = 2.0
        calls_made = 1

        # Main (1+1)-ES phase
        for _ in range(self.budget - 1):
            if time.time() > deadline:
                break
            z = rng.standard_normal(self.dim)
            y = np.clip(x + sigma * z, -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                x, fx = y, fy
                if fx < best_f:
                    best_f, best_x = fx, x.copy()
                sigma = min(sigma * 1.2, 10.0)
            else:
                sigma = max(sigma * 0.9, 1e-6)

        # Final high-precision polish pass
        x = best_x.copy()
        fx = best_f
        sigma = 0.1
        remaining = self.budget - calls_made
        if remaining > 0:
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                z = rng.standard_normal(self.dim)
                y = np.clip(x + sigma * z, -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                    sigma = min(sigma * 1.1, 1.0)
                else:
                    sigma = max(sigma * 0.85, 1e-8)

        return best_x, best_f
# EVOLVE-BLOCK-END
