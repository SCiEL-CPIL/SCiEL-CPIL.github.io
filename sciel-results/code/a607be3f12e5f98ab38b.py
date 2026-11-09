"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive step size and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit
        rng = np.random.default_rng(42)
        n = self.dim
        total_budget = self.budget
        calls_made = 0

        best_x = None
        best_f = np.inf

        # Multi-start strategy: split budget across several restarts
        n_restarts = 6
        budget_per_restart = total_budget // n_restarts
        last_restart_budget = total_budget - budget_per_restart * (n_restarts - 1)

        for r in range(n_restarts):
            if calls_made >= total_budget:
                break
            if time.time() > deadline:
                break

            # Initialize: first restart at origin, others random or near best
            if r == 0:
                x = np.zeros(n)
            elif best_x is not None and rng.random() < 0.4:
                # Sample near the best point found with moderate spread
                x = best_x + rng.normal(0, 1.5, n)
                x = np.clip(x, -5, 5)
            else:
                x = rng.uniform(-5, 5, n)

            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 2.0
            successes = 0
            window = 10
            window_count = 0

            remaining = min(budget_per_restart if r < n_restarts - 1 else last_restart_budget,
                           total_budget - calls_made)

            for _ in range(remaining):
                if calls_made >= total_budget:
                    break
                if time.time() > deadline:
                    break

                y = x + sigma * rng.standard_normal(n)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    successes += 1
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.3
                else:
                    sigma *= 0.9

                window_count += 1
                if window_count >= window:
                    # 1/5 success rule for global step size adjustment
                    if successes / window_count > 0.25:
                        sigma *= 1.2
                    elif successes / window_count < 0.15:
                        sigma *= 0.8
                    successes = 0
                    window_count = 0

                # Clamp sigma to reasonable bounds
                sigma = max(0.005, min(sigma, 10.0))

            # End of restart

        return best_x, best_f
# EVOLVE-BLOCK-END
