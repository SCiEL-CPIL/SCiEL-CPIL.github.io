"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with stagnation-based restarts, time guard, and final polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)

        # Initialize
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        sigma = 2.0
        calls = 1
        best_x = x.copy()
        best_fx = fx
        fail_count = 0
        stagnation_threshold = 30

        for _ in range(budget - 1):
            if time.time() > deadline:
                break

            # Generate offspring
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                # Accept
                x, fx = y, fy
                sigma *= 1.5
                fail_count = 0

                # Update best
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
            else:
                # Reject
                sigma *= 0.87
                fail_count += 1

                # Stagnation restart
                if fail_count >= stagnation_threshold:
                    # Restart from a new random point
                    x = rng.uniform(-5, 5, dim)
                    fx = func(x)
                    calls += 1
                    sigma = 2.0
                    fail_count = 0

                    # Update best if this random restart is better
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()

                    # If we used an extra call for the restart,
                    # we effectively consumed 2 calls in this iteration loop step.
                    if calls >= budget:
                        break

        # Final polish pass: tighten sigma and do a few more iterations if budget allows
        if calls < budget and time.time() < deadline:
            sigma = 0.1
            x = best_x.copy()
            fx = best_fx
            for _ in range(min(budget - calls, 50)):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.8

        return best_x, best_fx
# EVOLVE-BLOCK-END
