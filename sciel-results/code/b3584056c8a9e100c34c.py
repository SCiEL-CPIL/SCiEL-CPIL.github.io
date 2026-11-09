"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, deadline guard, and final polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 280  # safety margin before 300s kill
        rng = np.random.default_rng(0)
        d = self.dim
        budget = self.budget

        # Initial point
        x = rng.uniform(-5, 5, d)
        fx = func(x)
        best_x, best_fx = x.copy(), fx

        sigma = 2.0
        calls = 1

        def check_deadline():
            return time.time() >= deadline

        # Main (1+1)-ES loop
        while calls < budget:
            if check_deadline():
                break
            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                if fx < best_fx:
                    best_x, best_fx = x.copy(), fx
                sigma *= 1.5
            else:
                sigma *= 0.87

            # Early termination if converged
            if sigma < 1e-10:
                break

        # Final polish: local search with very small steps around best_x
        if calls < budget and not check_deadline():
            sigma_p = 1e-4
            for _ in range(min(50, budget - calls)):
                if check_deadline():
                    break
                y = np.clip(best_x + sigma_p * rng.standard_normal(d), -5, 5)
                fy = func(y)
                calls += 1
                if fy < best_fx:
                    best_x, best_fx = y, fy
                else:
                    sigma_p *= 0.7

        return best_x, best_fx
# EVOLVE-BLOCK-END
