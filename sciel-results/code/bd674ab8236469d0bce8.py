"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # 5s safety margin before 300s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        sigma = 1.0
        calls = 1

        # 1/5 success rule: maintain success rate around 0.2
        success = 0
        window = 10

        while calls < budget:
            if time.time() > deadline:
                break

            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x = y
                fx = fy
                success += 1
                if fy < best_fx:
                    best_x = x.copy()
                    best_fx = fy
            else:
                pass

            # Adjust sigma every 'window' steps
            if success >= window:
                sigma *= 1.25
                success = 0
                window = 10
            elif window <= 10:
                # Only shrink if we've tried enough
                if success < window * 0.2 and window >= 10:
                    sigma *= 0.8
                    window = 10
                else:
                    window += 1

            # Prevent sigma from getting too small or too large
            sigma = np.clip(sigma, 0.01, 5.0)

        return best_x, best_fx
# EVOLVE-BLOCK-END
