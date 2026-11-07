"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with adaptive step size and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit, 20s safety margin
        rng = np.random.default_rng(12345)

        dim = self.dim
        budget = self.budget

        # Initial point: center of the box (sphere minimum is at origin before shift,
        # but shift is hidden; starting at 0 is a reasonable prior)
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        # Adaptive step size
        sigma = 2.0
        successes = 0
        total = 0

        # Population-based CMA-ES variant
        lam = 4  # population size
        # We'll do a simple (1+1)-ES with adaptive sigma for reliability,
        # but with a population-based approach for better exploration.

        # Actually, let's use a proper CMA-ES-like approach with a small population
        # but keep it simple and robust.

        # Simple (1+1)-ES with 1/5 rule and time guard
        for i in range(budget - 1):
            # Check time
            if time.time() > deadline:
                break

            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)

            if fy < fx:
                x = y
                fx = fy
                successes += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                pass
            total += 1

            # Adaptive sigma: 1/5 success rule
            # Check every 5 steps
            if total % 5 == 0:
                if successes / 5 > 0.2:
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                successes = 0

            # Ensure sigma stays reasonable
            sigma = np.clip(sigma, 1e-8, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
