"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES-like optimizer with 1/5 success rule for Gallagher 101 peaks."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        total_budget = self.budget

        # Track best solution globally
        best_x = None
        best_f = np.inf

        # Multi-restart strategy: distribute budget across restarts
        # Based on knowledge: 5 restarts improved score from 1.0 to 1.143
        n_restarts = 5
        budget_per_restart = max(1, total_budget // n_restarts)

        # For the first few restarts, use more budget to explore
        # Later restarts use remaining budget
        restart_budgets = []
        remaining = total_budget
        for i in range(n_restarts):
            if i == n_restarts - 1:
                restart_budgets.append(remaining)
            else:
                b = budget_per_restart
                restart_budgets.append(b)
                remaining -= b

        # Ensure we don't exceed total budget
        total_used = sum(restart_budgets)
        if total_used > total_budget:
            restart_budgets[-1] = max(1, total_budget - sum(restart_budgets[:-1]))

        for restart_idx in range(n_restarts):
            if time.time() > deadline:
                break

            rb = restart_budgets[restart_idx]
            if rb < 1:
                rb = 1

            # Initialize with a random point, biased toward center for later restarts
            if restart_idx == 0:
                x = rng.uniform(-5, 5, dim)
            else:
                # Later restarts: mix of random and perturbation of best found
                if best_x is not None and rng.random() < 0.3:
                    x = np.clip(best_x + rng.standard_normal(dim) * 2.0, -5, 5)
                else:
                    x = rng.uniform(-5, 5, dim)

            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 2.0
            calls_in_restart = 1

            for _ in range(rb - 1):
                if time.time() > deadline:
                    break

                # Generate candidate
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_in_restart += 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.3
                else:
                    sigma *= 0.85

                # Adaptive sigma: prevent it from getting too small or too large
                sigma = np.clip(sigma, 0.01, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
