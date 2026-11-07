"""Rewritten optimizer for BBOB f17 (Schaffers F7) with restarts and adaptive ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Restart-based (1+1)-ES with adaptive sigma and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0
        remaining = self.budget

        # Number of restarts: use multiple restarts to escape local optima
        # Schaffers F7 has many local minima, so restarts are important
        n_restarts = 10
        budget_per_restart = max(1, self.budget // n_restarts)

        for r in range(n_restarts):
            if calls_made >= self.budget:
                break
            if time.time() > deadline - 5.0:
                break

            # Initialize: for first restart, try center; for others, random
            if r == 0:
                x = np.zeros(self.dim)
            else:
                x = rng.uniform(-5, 5, self.dim)

            fx = func(x)
            calls_made += 1
            remaining = self.budget - calls_made

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Adaptive (1+1)-ES within this restart
            sigma = 2.0
            success_count = 0
            trials = 0

            while remaining > 0:
                if time.time() > deadline - 3.0:
                    break

                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1
                remaining = self.budget - calls_made
                trials += 1

                if fy < fx:
                    x = y
                    fx = fy
                    success_count += 1
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9

                # Adaptive sigma adjustment based on success rate
                if trials >= 10:
                    rate = success_count / trials
                    if rate > 0.3:
                        sigma *= 1.05
                    elif rate < 0.1:
                        sigma *= 0.95
                    success_count = 0
                    trials = 0

                # Early stopping if sigma becomes too small
                if sigma < 1e-6:
                    break

                # Check time
                if time.time() > deadline - 3.0:
                    break

            # If we used up budget, break
            if calls_made >= self.budget:
                break

        # Ensure at least one call was made
        if calls_made == 0:
            x = np.zeros(self.dim)
            fx = func(x)
            best_x = x
            best_f = fx

        return best_x, best_f
# EVOLVE-BLOCK-END
