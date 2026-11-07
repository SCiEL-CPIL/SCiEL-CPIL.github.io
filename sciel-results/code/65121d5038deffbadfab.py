"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, time guard, and periodic random restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 290.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        sigma = 2.0
        n_calls = 1

        for _ in range(budget - 1):
            if time.time() > deadline:
                break

            # Periodic random restart if sigma has shrunk too much (stall detection)
            if n_calls % 30 == 0 and sigma < 0.05:
                # Try a random restart
                xr = rng.uniform(-4, 4, dim)
                fr = func(xr)
                n_calls += 1
                if fr < fx:
                    x, fx = xr, fr
                    if fx < best_f:
                        best_x, best_f = x.copy(), fx
                    sigma = 2.0
                else:
                    # If restart doesn't help, reduce sigma further to try to escape
                    sigma *= 0.5
                    if sigma < 1e-6:
                        # If sigma is too small, do a bigger random jump
                        xr = rng.uniform(-5, 5, dim)
                        fr = func(xr)
                        n_calls += 1
                        if fr < fx:
                            x, fx = xr, fr
                            if fx < best_f:
                                best_x, best_f = x.copy(), fx
                            sigma = 1.0
                        else:
                            sigma = 0.1
                continue

            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            n_calls += 1

            if fy < fx:
                x, fx = y, fy
                if fx < best_f:
                    best_x, best_f = x.copy(), fx
                sigma *= 1.5
            else:
                sigma *= 0.87

        return best_x, best_f
# EVOLVE-BLOCK-END
