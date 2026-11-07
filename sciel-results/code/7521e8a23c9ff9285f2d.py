"""Optimizer for BBOB-style f6 (Attractive sector)."""
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with multi-start restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_fx = np.inf

        # We'll use a multi-start strategy:
        # - Run (1+1)-ES segments
        # - If stagnation is detected, restart from a new random point or near the best
        # - Track global best across all segments

        # Initial random start
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        calls = 1

        if fx < best_fx:
            best_x, best_fx = x.copy(), fx

        sigma = 2.0
        no_improve_count = 0
        stagnation_threshold = 25  # restart after 25 consecutive non-improvements

        # We'll also track a "restart budget" - how many restarts we've done
        # and allocate remaining budget accordingly
        restarts_done = 0
        max_restarts = 8

        while calls < budget and restarts_done < max_restarts:
            # Check time
            if time.time() > deadline:
                break

            # Run a segment of (1+1)-ES
            segment_calls = 0
            max_segment_calls = 60  # max calls per segment before forced restart

            while calls < budget and segment_calls < max_segment_calls:
                if time.time() > deadline:
                    break

                # Adaptive step size
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                segment_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2  # increase step on success
                    no_improve_count = 0
                    if fx < best_fx:
                        best_x, best_fx = x.copy(), fx
                else:
                    sigma *= 0.85  # decrease step on failure
                    no_improve_count += 1

                # Restart if stagnating
                if no_improve_count >= stagnation_threshold:
                    break

                if calls >= budget:
                    break

            # If we broke out due to stagnation or segment limit, prepare restart
            if calls >= budget or time.time() > deadline:
                break

            restarts_done += 1

            # Choose restart strategy:
            # 1. With 50% probability, restart near the global best with moderate perturbation
            # 2. With 50% probability, start completely fresh from a random point
            if rng.random() < 0.5 and best_x is not None:
                # Restart near best with perturbation
                x = np.clip(best_x + 1.5 * rng.standard_normal(dim), -5, 5)
            else:
                # Fresh random start
                x = rng.uniform(-5, 5, dim)

            fx = func(x)
            calls += 1

            if fx < best_fx:
                best_x, best_fx = x.copy(), fx

            # Reset parameters for new segment
            sigma = 2.0
            no_improve_count = 0

        # Final polish: if we have budget remaining, do a fine-tuning pass near the best
        if calls < budget and best_x is not None and time.time() < deadline:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.1
            no_improve_count = 0
            polish_calls = 0
            max_polish_calls = 30

            while calls < budget and polish_calls < max_polish_calls:
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                polish_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 0.95
                    no_improve_count = 0
                    if fx < best_fx:
                        best_x, best_fx = x.copy(), fx
                else:
                    sigma *= 0.8
                    no_improve_count += 1

                if no_improve_count >= 15:
                    break

        return best_x, best_fx
