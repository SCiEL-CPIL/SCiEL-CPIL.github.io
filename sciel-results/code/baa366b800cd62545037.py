"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive step size and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit
        rng = np.random.default_rng(42)
        d = self.dim
        total_budget = self.budget

        # We'll track the global best
        best_x = None
        best_f = np.inf

        # Multi-restart strategy: divide budget among restarts
        # Use more restarts for early exploration, fewer for exploitation
        # For 300 budget, use about 5-10 restarts
        n_restarts = 8
        budget_per_restart = total_budget // n_restarts
        remaining = total_budget - budget_per_restart * n_restarts
        # Give extra to first few restarts for exploration

        calls_made = 0

        for r in range(n_restarts):
            # Check time
            if time.time() > deadline:
                break
            # Check budget
            if calls_made >= total_budget:
                break

            # Initialize: first restart at origin-ish, others random
            if r == 0:
                x = np.zeros(d)
            else:
                x = rng.uniform(-5, 5, d)

            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 2.0 if r == 0 else 1.5
            # Adaptive sigma with success counter
            successes = 0
            window = 10

            # Determine budget for this restart
            if r < n_restarts - 1:
                b_r = budget_per_restart
            else:
                b_r = total_budget - calls_made

            for i in range(b_r):
                # Time guard
                if time.time() > deadline:
                    break
                # Budget guard
                if calls_made >= total_budget:
                    break

                # Generate candidate
                y = x + sigma * rng.standard_normal(d)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    # Increase sigma on success
                    sigma *= 1.2
                    if sigma > 5.0:
                        sigma = 5.0
                else:
                    # Decrease sigma on failure
                    sigma *= 0.85
                    if sigma < 0.01:
                        sigma = 0.01

                # Periodic sigma adaptation based on success rate
                if (i + 1) % window == 0:
                    success_rate = successes / window
                    if success_rate > 0.5:
                        sigma *= 1.1
                    elif success_rate < 0.3:
                        sigma *= 0.9
                    successes = 0

                # Early termination if sigma is very small and no progress
                if sigma < 0.05 and i > 20:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
