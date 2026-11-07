"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Multi-start (1+1)-ES with adaptive step size and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit
        budget = self.budget
        dim = self.dim
        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        calls = 0

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Phase 1: Multi-start to find a good basin
        # Use a few random restarts to explore the landscape
        num_restarts = 5
        calls_per_restart = max(1, (budget - 1) // (num_restarts + 1))
        # Reserve some budget for fine-tuning the best solution

        for r in range(num_restarts):
            if calls >= budget:
                break
            if time.time() > deadline - 5.0:
                break

            # Random starting point
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls += 1
            update_best(x, fx)

            sigma = 3.0  # Large initial step to explore
            # Local (1+1)-ES for this restart
            max_iter = calls_per_restart
            for i in range(max_iter):
                if calls >= budget:
                    break
                if time.time() > deadline - 5.0:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2  # Increase step on success
                    update_best(x, fx)
                else:
                    sigma *= 0.8  # Decrease step on failure

                    # If sigma becomes too small, break early and try next restart
                    if sigma < 1e-6:
                        break

        # Phase 2: Fine-tune the best solution found
        if best_x is not None and calls < budget:
            x = best_x
            fx = best_f
            sigma = 0.5  # Start with a moderate step for fine-tuning

            # Use remaining budget for fine-tuning
            while calls < budget:
                if time.time() > deadline - 1.0:
                    break

                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                    update_best(x, fx)
                else:
                    sigma *= 0.9

                    # If we've made no progress in a while, reduce sigma more aggressively
                    if sigma < 1e-8:
                        break

        # Ensure we have a valid result
        if best_x is None:
            # Fallback: evaluate origin
            x = np.zeros(dim)
            fx = func(x)
            calls += 1
            best_x = x
            best_f = fx

        return best_x, best_f
# EVOLVE-BLOCK-END
