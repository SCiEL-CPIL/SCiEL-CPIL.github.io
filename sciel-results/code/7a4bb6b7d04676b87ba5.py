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

        # Initialize: start near center (optimum lies in [-4,4]^5)
        x = rng.uniform(-2, 2, dim)
        fx = func(x)
        sigma = 2.0
        calls = 1
        best_x = x.copy()
        best_fx = fx
        fail_count = 0
        stagnation_threshold = 30
        restart_count = 0

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
                    restart_count += 1
                    # Center-biased for first few restarts, then uniform
                    if restart_count <= 3:
                        x = rng.uniform(-3, 3, dim)
                    else:
                        x = rng.uniform(-5, 5, dim)
                    fx = func(x)
                    calls += 1
                    sigma = 2.0
                    fail_count = 0

                    # Update best if this random restart is better
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()

                    if calls >= budget:
                        break

        # Final polish pass: tighten sigma and do a few more iterations if budget allows
        if calls < budget and time.time() < deadline:
            sigma = 0.1
            x = best_x.copy()
            fx = best_fx
            polish_budget = min(budget - calls, 50)
            for _ in range(polish_budget):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85
                    sigma = max(sigma, 1e-5)

        return best_x, best_fx
# EVOLVE-BLOCK-END
