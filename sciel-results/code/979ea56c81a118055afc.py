"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, restarts, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        n_calls = 0
        total_budget = self.budget

        def _time_left():
            return time.time() < deadline

        # Best solution found so far
        best_x = None
        best_f = np.inf

        # We'll do a few restarts, each with a portion of the budget
        # Use CMA-ES-like adaptive step size with (1+1)-ES
        # Strategy: multiple restarts with different initial points

        num_restarts = 5
        budget_per_restart = total_budget // num_restarts

        for restart_idx in range(num_restarts):
            if n_calls >= total_budget:
                break
            if not _time_left():
                break

            # Initialize: first restart at origin, others random
            if restart_idx == 0:
                x = np.zeros(self.dim)
            else:
                x = rng.uniform(-4, 4, self.dim)

            fx = func(x)
            n_calls += 1

            if fx < best_f:
                best_f, best_x = fx, x.copy()

            sigma = 1.0
            # Adaptive step size: increase on success, decrease on failure
            # Use a more aggressive strategy
            steps = 0
            max_steps = budget_per_restart

            for _ in range(max_steps):
                if n_calls >= total_budget:
                    break
                if not _time_left():
                    break

                # Generate candidate
                direction = rng.standard_normal(self.dim)
                y = np.clip(x + sigma * direction, -5, 5)
                fy = func(y)
                n_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3  # increase step size on success
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                    steps += 1
                else:
                    sigma *= 0.8  # decrease step size on failure
                    # If sigma gets too small, break early for this restart
                    if sigma < 1e-8:
                        break

            # If we used up most of our budget, stop
            if n_calls >= total_budget:
                break

        # Final polish: if we have budget left, do fine-tuning around best
        remaining = total_budget - n_calls
        if remaining > 10 and best_x is not None and _time_left():
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for _ in range(remaining):
                if not _time_left():
                    break
                direction = rng.standard_normal(self.dim)
                y = np.clip(x + sigma * direction, -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f, best_x = fx, x.copy()
                else:
                    sigma *= 0.85
                    if sigma < 1e-10:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
