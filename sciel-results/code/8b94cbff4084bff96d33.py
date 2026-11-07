"""Optimizer for BBOB-style f4 (Büche-Rastrigin) function."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, smart restarts, and polish phase."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Global best tracking
        global_best_x = np.zeros(dim)
        global_best_fx = func(global_best_x)
        calls_made = 1

        # (1+1)-ES parameters with 1/5 success rule
        sigma = 2.0
        min_sigma = 1e-6
        max_sigma = 6.0
        success_factor = 1.5
        failure_factor = 0.87
        stagnation_limit = 25

        # Main search phase: use ~80% of budget
        main_budget = int(budget * 0.80)
        polish_budget = budget - main_budget

        while calls_made < main_budget:
            if time.time() > deadline:
                break

            # Generate candidate
            y = global_best_x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < global_best_fx:
                global_best_x = y
                global_best_fx = fy
                sigma = min(sigma * success_factor, max_sigma)
            else:
                sigma = max(sigma * failure_factor, min_sigma)

            # Stagnation / restart check
            # If sigma is very small, we're likely stuck in a local minimum
            if sigma < 1e-4 or (sigma < 0.1 and calls_made > 50):
                # Restart from global best with a fresh, larger sigma
                sigma = 3.0
                # Optionally nudge the starting point slightly
                global_best_x = np.clip(global_best_x + 0.5 * rng.standard_normal(dim), -5, 5)
                global_best_fx = func(global_best_x)
                calls_made += 1

                if calls_made >= main_budget:
                    break

        # Polish phase: fine-grained search around the global best
        if calls_made < budget:
            sigma = 0.05
            polish_start_x = global_best_x.copy()
            polish_start_fx = global_best_fx
            polish_stagnation = 0

            while calls_made < budget:
                if time.time() > deadline:
                    break

                y = polish_start_x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < polish_start_fx:
                    polish_start_x = y
                    polish_start_fx = fy
                    sigma = min(sigma * 1.2, 0.1)
                    polish_stagnation = 0

                    if fy < global_best_fx:
                        global_best_fx = fy
                        global_best_x = y
                else:
                    sigma = max(sigma * 0.9, 1e-8)
                    polish_stagnation += 1

                if polish_stagnation > 20:
                    # Break out of polish if stuck
                    break

        return global_best_x, global_best_fx
# EVOLVE-BLOCK-END
