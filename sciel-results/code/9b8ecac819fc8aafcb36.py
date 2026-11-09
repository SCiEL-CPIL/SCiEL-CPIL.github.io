"""Optimizer for BBOB-style f24 (Lunacek bi-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and multiple restarts for BBOB f24."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # 15s safety margin before 300s limit

        n = self.dim
        total_budget = self.budget

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        # 5 restarts with geometrically decreasing budgets
        # More restarts help escape the deceptive local minimum of Lunacek
        num_restarts = 5
        # Budgets: 35%, 25%, 20%, 12%, 8%
        budgets = [
            int(total_budget * 0.35),
            int(total_budget * 0.25),
            int(total_budget * 0.20),
            int(total_budget * 0.12),
            total_budget - int(total_budget * 0.35) - int(total_budget * 0.25) - int(total_budget * 0.20) - int(total_budget * 0.12)
        ]

        calls_made = 0

        for restart_idx in range(num_restarts):
            if calls_made >= total_budget:
                break
            if time.time() > deadline:
                break

            budget_this_restart = min(budgets[restart_idx], total_budget - calls_made)
            if budget_this_restart < 5:
                break

            # Different initialization strategies for each restart
            if restart_idx == 0:
                x = np.zeros(n)
            elif restart_idx == 1:
                x = rng.uniform(-3, 3, n)
            elif restart_idx == 2:
                x = rng.uniform(-5, 5, n)
            elif restart_idx == 3:
                if best_x is not None:
                    x = np.clip(best_x + rng.standard_normal(n) * 1.5, -5, 5)
                else:
                    x = rng.uniform(-5, 5, n)
            else:
                if best_x is not None:
                    x = np.clip(best_x + rng.standard_normal(n) * 2.5, -5, 5)
                else:
                    x = rng.uniform(-5, 5, n)

            fx = func(x)
            calls_made += 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.2  # Initial step size

            # 1/5 success rule parameters
            window = 10
            successes = 0
            window_count = 0

            # Main (1+1)-ES loop with 1/5 success rule
            while calls_made < total_budget:
                if time.time() > deadline:
                    break

                # Check if we've used up this restart's budget
                used_in_restart = calls_made - (total_budget - budget_this_restart)
                if used_in_restart >= budget_this_restart:
                    break

                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

                window_count += 1

                # Adapt sigma using 1/5 success rule every `window` steps
                if window_count >= window:
                    rate = successes / window_count
                    if rate > 0.2:
                        sigma *= 1.25
                    elif rate < 0.15:
                        sigma *= 0.82
                    successes = 0
                    window_count = 0

                # Clamp sigma
                sigma = np.clip(sigma, 1e-8, 8.0)

        # Ensure we return valid values
        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls_made += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
