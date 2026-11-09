import numpy as np
import time

class Optimizer:
    """(1+1)-ES with 1/5 success rule, seeded for reproducibility."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = float(func(x))
        best_x = x.copy()
        best_f = fx

        sigma = 2.0

        # Track success rate for 1/5 rule
        successes = 0
        window = 0
        window_size = 15  # typical window for 1/5 rule

        for i in range(budget - 1):
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = float(func(y))

            if fy < fx:
                x, fx = y, fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                successes += 1
            else:
                successes = 0

            window += 1

            # Adapt sigma using 1/5 success rule
            if window >= window_size:
                success_rate = successes / window
                if success_rate > 0.2:
                    sigma *= 1.5
                elif success_rate < 0.2:
                    sigma *= 0.87
                successes = 0
                window = 0

            # Safety: if sigma becomes too small, we're likely converged
            if sigma < 1e-8:
                break

        return best_x, best_f
