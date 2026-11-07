"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, multi-restart, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Global best
        global_x = None
        global_fx = np.inf

        # Number of restarts
        max_restarts = 20
        calls_used = 0

        for restart in range(max_restarts):
            if calls_used >= budget:
                break
            if time.time() > deadline:
                break

            # Initialize
            if restart == 0:
                rng = np.random.default_rng(0)
                x = rng.uniform(-3, 3, dim)
            else:
                rng = np.random.default_rng(restart * 7919 + 13)
                if global_x is not None and restart < 10:
                    # Perturb global best with increasing noise
                    noise_scale = 2.0 * (restart / 10.0) + 1.0
                    x = np.clip(global_x + noise_scale * rng.standard_normal(dim), -5, 5)
                else:
                    x = rng.uniform(-5, 5, dim)

            fx = func(x)
            calls_used += 1
            if fx < global_fx:
                global_fx = fx
                global_x = x.copy()

            sigma = 1.0
            # Adaptive step size with 1/5 rule
            window = 10
            successes = 0
            total = 0

            # Run (1+1)-ES until no improvement for a while or budget/time exhausted
            no_improve_count = 0
            max_no_improve = 50 + 10 * restart  # increase patience slightly

            while calls_used < budget:
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_used += 1

                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    no_improve_count = 0
                    if fx < global_fx:
                        global_fx = fx
                        global_x = x.copy()
                else:
                    no_improve_count += 1

                total += 1
                if total >= window:
                    rate = successes / total
                    if rate > 0.2:
                        sigma *= 1.3
                    elif rate < 0.15:
                        sigma *= 0.7
                    successes = 0
                    total = 0

                # Clamp sigma
                sigma = np.clip(sigma, 0.01, 10.0)

                if no_improve_count >= max_no_improve:
                    break

        return global_x, global_fx
# EVOLVE-BLOCK-END
