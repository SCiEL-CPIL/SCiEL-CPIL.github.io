"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with diverse restarts and polish phase."""

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

        # Phase 1: Multi-start exploration with (1+1)-ES
        # Use most of the budget for exploration with restarts
        polish_budget = max(30, budget // 20)
        search_budget = budget - polish_budget

        # Define diverse starting points for restarts
        start_points = [
            np.zeros(d),
            rng.uniform(-4.0, 4.0, d),
            rng.uniform(-3.0, 3.0, d),
            rng.uniform(-4.5, 4.5, d),
            rng.uniform(-2.0, 2.0, d),
            rng.uniform(-4.0, 4.0, d),
        ]

        # Evaluate initial points and set up search
        x = start_points[0].copy()
        fx = func(x)
        total_calls += 1
        update_best(x, fx)

        sigma = 2.5
        no_improve = 0
        max_no_improve = 40
        restart_idx = 1
        max_restarts = len(start_points) - 1

        while total_calls < search_budget and not check_deadline():
            y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
            fy = func(y)
            total_calls += 1

            if fy < fx:
                x, fx = y, fy
                no_improve = 0
                sigma *= 1.4
                update_best(x, fx)
            else:
                sigma *= 0.88
                no_improve += 1

            # Restart if stalled and we have more start points
            if no_improve >= max_no_improve and restart_idx < max_restarts:
                restart_idx += 1
                x = start_points[restart_idx].copy()
                fx = func(x)
                total_calls += 1
                update_best(x, fx)
                sigma = 2.0
                no_improve = 0
            elif no_improve >= max_no_improve * 2:
                # If we've exhausted all restarts, try a fresh random restart
                x = rng.uniform(-4.5, 4.5, d)
                fx = func(x)
                total_calls += 1
                update_best(x, fx)
                sigma = 1.5
                no_improve = 0

        # Phase 2: Polish around the best point found
        if best_x is not None and not check_deadline():
            x = best_x.copy()
            fx = best_f
            sigma = 0.5

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
                    sigma *= 1.05
                else:
                    sigma *= 0.92

                if sigma < 1e-9:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
