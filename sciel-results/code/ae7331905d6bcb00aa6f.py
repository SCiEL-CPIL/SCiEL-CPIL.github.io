"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, restarts, and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        rng = np.random.default_rng(0)

        # Best solution found overall
        best_x = None
        best_f = np.inf

        # Total evaluations used
        evals_used = 0

        # We'll do multiple restarts, each with a portion of the budget
        # Strategy: use several restarts to explore different regions
        # For 5D/300 budget, let's do restarts of ~50-80 evals each

        restart_budgets = [50, 50, 50, 50, 50, 50]  # 6 restarts of 50 = 300
        # But we need to track total budget

        total_budget = self.budget
        remaining = total_budget

        for restart_idx, rb in enumerate(restart_budgets):
            if remaining <= 0:
                break
            if time.time() > deadline - 5.0:
                break

            rb = min(rb, remaining)
            if rb < 5:
                break

            # Initialize: use best found so far or random
            if best_x is not None and rng.random() < 0.3:
                # Perturb around best
                x = np.clip(best_x + rng.standard_normal(self.dim) * 0.5, -5, 5)
            else:
                x = rng.uniform(-5, 5, self.dim)

            fx = func(x)
            evals_used += 1
            remaining -= 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 2.0
            # 1/5 success rule parameters
            sigma_min = 1e-10
            sigma_max = 10.0

            for i in range(rb - 1):
                if remaining <= 0:
                    break
                if time.time() > deadline - 5.0:
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                evals_used += 1
                remaining -= 1

                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma = min(sigma * 1.2, sigma_max)
                else:
                    sigma = max(sigma * 0.85, sigma_min)

                # Early termination if sigma is very small
                if sigma < 1e-8:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
