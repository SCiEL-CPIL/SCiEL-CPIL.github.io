"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with aggressive sigma adaptation and lightweight random restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        d = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        total_calls = 0

        def check_deadline():
            return time.time() > deadline or total_calls >= budget

        def update_best(x, fx):
            nonlocal best_x, best_f
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 1: Main (1+1)-ES with aggressive adaptation and random restarts
        # Allocate most of the budget to the main search
        polish_budget = max(20, budget // 25)
        search_budget = budget - polish_budget

        # Start at the origin, which is a standard and often effective starting point
        x = np.zeros(d)
        fx = func(x)
        total_calls += 1
        update_best(x, fx)

        sigma = 2.0
        no_improve = 0
        max_no_improve = 60  # Threshold for restart
        restart_count = 0
        max_restarts = 6     # Limit number of random restarts to avoid wasting budget

        while total_calls < search_budget and not check_deadline():
            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)
            total_calls += 1

            if fy < fx:
                # Success: move to new point, increase sigma aggressively
                x, fx = y, fy
                no_improve = 0
                sigma *= 1.5
                update_best(x, fx)
            else:
                # Failure: stay at current point, decrease sigma
                sigma *= 0.87
                no_improve += 1

            # Lightweight restart: if stalled, jump to a new random point
            if no_improve >= max_no_improve and restart_count < max_restarts:
                restart_count += 1
                x = rng.uniform(-4.5, 4.5, d)
                fx = func(x)
                total_calls += 1
                update_best(x, fx)
                sigma = 1.5  # Reset sigma to a moderate value for new search
                no_improve = 0

            # If we've used all restarts and still stalled, just continue shrinking sigma
            # This allows for fine-grained search in the current basin if no restart is left

        # Phase 2: Final polish around the best point found
        if best_x is not None and not check_deadline():
            x = best_x.copy()
            fx = best_f
            sigma = 0.3  # Start polish with a smaller step size

            remaining_calls = budget - total_calls
            polish_steps = min(polish_budget, remaining_calls)

            for _ in range(polish_steps):
                if check_deadline():
                    break

                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                total_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma *= 1.1  # Slight increase on success in polish
                else:
                    sigma *= 0.9   # Decrease on failure in polish

                # Stop if sigma becomes too small
                if sigma < 1e-8:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
