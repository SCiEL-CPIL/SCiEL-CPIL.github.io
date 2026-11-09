"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive step size and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Start at center (often good for rotated Rosenbrock)
        x = np.zeros(n)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls_used = 1

        sigma = 3.0
        no_improve = 0
        max_no_improve = 30

        while calls_used < budget:
            # Check time budget
            if time.time() > deadline:
                break

            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)
            calls_used += 1

            if fy < fx:
                # Accept step
                x = y
                fx = fy
                no_improve = 0
                sigma *= 1.2

                # Update global best
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            else:
                # Reject step
                no_improve += 1
                sigma *= 0.8

                # If stuck, try a restart from a random point
                if no_improve >= max_no_improve:
                    no_improve = 0
                    # Try a few random restarts
                    for _ in range(5):
                        if calls_used >= budget or time.time() > deadline:
                            break
                        x_r = rng.uniform(-5, 5, n)
                        fx_r = func(x_r)
                        calls_used += 1
                        if fx_r < best_f:
                            best_f = fx_r
                            best_x = x_r.copy()
                            x = x_r
                            fx = fx_r
                            sigma = 3.0
                            break
                    else:
                        # No improvement from restarts, keep searching from current x
                        sigma = 3.0

        return best_x, best_f
# EVOLVE-BLOCK-END
