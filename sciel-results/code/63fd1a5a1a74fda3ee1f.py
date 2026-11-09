"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with multiple restarts for BBOB f24 (Lunacek bi-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit

        n = self.dim
        total_budget = self.budget

        # Fixed seed for reproducibility
        rng = np.random.default_rng(0)

        # Best solution found so far across all restarts
        best_x = None
        best_f = np.inf

        # Multiple restarts to escape local minima of Lunacek bi-Rastrigin
        # Allocate budget across restarts. Use 3 restarts with decreasing budget.
        # First restart gets more budget since it's more likely to find the global minimum.
        num_restarts = 3
        # Budget allocation: 50%, 30%, 20%
        budgets = [int(total_budget * 0.5), int(total_budget * 0.3), total_budget - int(total_budget * 0.5) - int(total_budget * 0.3)]

        calls_made = 0

        for restart_idx in range(num_restarts):
            if calls_made >= total_budget:
                break
            if time.time() > deadline:
                break

            budget_this_restart = min(budgets[restart_idx], total_budget - calls_made)
            if budget_this_restart < 10:
                break

            # Initialize: use center for first restart, random for others
            if restart_idx == 0:
                x = np.zeros(n)
            else:
                x = rng.uniform(-5, 5, n)

            fx = func(x)
            calls_made += 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 2.0

            # Main (1+1)-ES loop for this restart
            while calls_made < total_budget and (total_budget - calls_made) > 0:
                if time.time() > deadline:
                    break

                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.5
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.87

                # Clamp sigma
                sigma = np.clip(sigma, 1e-8, 1e8)

                # Check if we've exhausted this restart's budget
                if calls_made - (total_budget - budget_this_restart) >= budget_this_restart:
                    break

        # Ensure we return valid values
        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
