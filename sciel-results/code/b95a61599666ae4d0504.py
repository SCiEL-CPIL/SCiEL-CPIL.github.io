"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with center start, adaptive sigma, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 270.0  # safety margin before 300s limit

        rng = np.random.default_rng(123)

        # Start at center: strong prior for sphere-like functions with shifts in [-4,4]
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls_made = 1

        sigma = 1.5
        no_improve = 0
        patience = 80  # restart after this many non-improving steps

        for _ in range(budget - 1):
            if time.time() >= deadline:
                break
            if calls_made >= budget:
                break

            # Generate candidate
            step = sigma * rng.standard_normal(dim)
            y = np.clip(x + step, -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                # Accept step
                x = y
                fx = fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                no_improve = 0
                # Increase sigma on success (geometric increase)
                sigma = min(sigma * 1.2, 10.0)
            else:
                no_improve += 1
                # Decrease sigma on failure
                sigma = max(sigma * 0.85, 0.01)

            # Restart if stuck
            if no_improve >= patience:
                # Try a few random restarts near best
                for _ in range(3):
                    if time.time() >= deadline or calls_made >= budget:
                        break
                    xr = best_x + 0.5 * rng.standard_normal(dim)
                    xr = np.clip(xr, -5, 5)
                    fr = func(xr)
                    calls_made += 1
                    if fr < fx:
                        x = xr
                        fx = fr
                        if fr < best_f:
                            best_f = fr
                            best_x = xr.copy()
                        no_improve = 0
                        sigma = 1.0
                        break
                else:
                    # No improvement from restarts, reduce sigma further
                    sigma = max(sigma * 0.5, 0.005)
                    no_improve = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
