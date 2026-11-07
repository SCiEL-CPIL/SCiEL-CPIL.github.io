"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive step size for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        dim = self.dim
        total_budget = self.budget

        # Track global best
        global_best_x = None
        global_best_fx = np.inf

        # Use multiple restarts with different initial points and seeds
        # For Büche-Rastrigin, multiple local minima exist, so restarts help
        n_restarts = 5
        budget_per_restart = total_budget // n_restarts

        for r in range(n_restarts):
            if time.time() > deadline:
                break

            # Different initial points for each restart
            rng = np.random.default_rng(100 + r * 7)

            # Start from different locations in the box
            if r == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-4, 4, size=dim)

            fx = func(x)

            if fx < global_best_fx:
                global_best_fx = fx
                global_best_x = x.copy()

            # Adaptive step size
            sigma = 1.0
            min_sigma = 1e-9
            max_sigma = 10.0

            calls_in_this_restart = 1

            # (1+1)-ES loop for this restart
            while calls_in_this_restart < budget_per_restart:
                if time.time() > deadline:
                    break

                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_in_this_restart += 1

                if fy < fx:
                    x, fx = y, fy
                    # Increase step size on success
                    sigma = min(sigma * 1.15, max_sigma)
                    if fx < global_best_fx:
                        global_best_fx = fx
                        global_best_x = x.copy()
                else:
                    # Decrease step size on failure
                    sigma = max(sigma * 0.85, min_sigma)

                    # If sigma gets too small, consider this restart done
                    if sigma < 1e-7:
                        break

        return global_best_x, global_best_fx
# EVOLVE-BLOCK-END
