"""Optimized BBOB f22 solver with deadline guard and adaptive multi-scale search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive stochastic search with multi-scale exploration and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Initialize with a few random starts to find a good basin
        x = np.zeros(self.dim)
        fx = np.inf

        # Try several initial points
        for _ in range(5):
            x_try = rng.uniform(-4, 4, self.dim)
            f_try = func(x_try)
            if f_try < fx:
                x, fx = x_try, f_try

        calls_made = 6  # 5 initial + 1 implicit

        # Adaptive step size
        sigma = 3.0
        best_x = x.copy()
        best_f = fx

        # Main search loop
        while calls_made < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate candidate
            direction = rng.standard_normal(self.dim)
            y = np.clip(x + sigma * direction, -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                x = y
                fx = fy
                if fx < best_f:
                    best_x, best_f = x.copy(), fx
                sigma *= 1.15
            else:
                sigma *= 0.92

            # Occasionally do a larger jump to escape local minima
            if calls_made % 50 == 0 and sigma < 0.5:
                y_jump = np.clip(x + rng.uniform(-2, 2, self.dim), -5, 5)
                fy_jump = func(y_jump)
                calls_made += 1
                if fy_jump < fx:
                    x, fx = y_jump, fy_jump
                    if fx < best_f:
                        best_x, best_f = x.copy(), fx
                    sigma = 1.0
                else:
                    sigma = max(sigma, 0.5)

        return best_x, best_f
# EVOLVE-BLOCK-END
