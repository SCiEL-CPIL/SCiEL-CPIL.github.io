"""Combined optimizer: LHS initialization + (1+1)-ES with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with LHS initialization and deadline guard for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Latin Hypercube-style sampling for fast early progress
        # Use a fraction of budget for exploration
        n_lhs = min(50, budget // 10)
        best_x = None
        best_f = np.inf

        for i in range(n_lhs):
            # LHS: divide each dimension into n_lhs equal intervals, pick one random point per interval
            x = np.zeros(dim)
            for d in range(dim):
                interval = i % n_lhs
                x[d] = -5.0 + (interval + rng.random()) * (10.0 / n_lhs)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Check deadline
            if time.time() > deadline:
                break

        # Phase 2: (1+1)-ES with restarts from the best point
        x = best_x.copy()
        fx = best_f
        sigma = 1.0
        calls_used = n_lhs
        no_improve_count = 0
        max_no_improve = 30
        sigma_min = 0.01
        sigma_max = 5.0

        while calls_used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate trial point
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls_used += 1

            if fy < fx:
                x, fx = y, fy
                sigma = min(sigma * 1.3, sigma_max)
                no_improve_count = 0
            else:
                sigma = max(sigma * 0.8, sigma_min)
                no_improve_count += 1

            # Restart mechanism: if stuck for too long, jump to a new random point
            # but keep the best found so far
            if no_improve_count >= max_no_improve:
                # Try a few random restarts
                for _ in range(3):
                    if calls_used >= budget or time.time() > deadline:
                        break
                    x_r = rng.uniform(-5, 5, dim)
                    fx_r = func(x_r)
                    calls_used += 1
                    if fx_r < fx:
                        x, fx = x_r, fx_r
                        sigma = 1.0
                        no_improve_count = 0
                        break
                else:
                    # If no restart helped, continue with smaller sigma
                    sigma = max(sigma * 0.5, sigma_min)
                    no_improve_count = 0

        return x, fx
# EVOLVE-BLOCK-END
