"""Black-box continuous optimizer for BBOB-style f9 (Rosenbrock rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive sigma and occasional restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        # Multi-restart (1+1)-ES
        max_restarts = 5
        calls_per_restart = self.budget // max_restarts

        for restart in range(max_restarts):
            if calls_made >= self.budget:
                break
            if time.time() > deadline - 10:  # need at least 10s remaining
                break

            # Initialize: if we have a previous best, perturb it; otherwise random
            if best_x is not None and restart > 0:
                x = np.clip(best_x + 0.5 * rng.standard_normal(self.dim), -5, 5)
            else:
                x = rng.uniform(-4, 4, self.dim)

            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f, best_x = fx, x

            sigma = 1.5
            no_improve = 0
            max_no_improve = 30

            remaining = self.budget - calls_made
            for i in range(remaining):
                if time.time() > deadline - 5:
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f, best_x = fx, x
                    sigma *= 1.2
                    no_improve = 0
                else:
                    sigma *= 0.85
                    no_improve += 1

                if no_improve >= max_no_improve:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
