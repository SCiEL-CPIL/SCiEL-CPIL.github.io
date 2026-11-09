"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with adaptive step size and restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Track best solution found
        best_x = np.zeros(n)
        best_f = np.inf
        call_count = 0

        def check_deadline():
            return time.time() > deadline

        # Strategy: CMA-ES with restarts
        # We'll use a simplified CMA-ES approach with:
        # - Population-based search
        # - Adaptive step size based on success rate
        # - Restart if stuck

        # Initial exploration: try several starting points
        num_starts = 3
        points_per_start = budget // num_starts

        for start_idx in range(num_starts):
            if call_count >= budget or check_deadline():
                break

            # Choose starting point
            if start_idx == 0:
                x = np.zeros(n)
            else:
                x = rng.uniform(-5, 5, n)

            fx = func(x)
            call_count += 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            if call_count >= budget or check_deadline():
                break

            # CMA-ES-like iteration from this starting point
            sigma = 2.0
            success_count = 0
            pop_size = 5
            window = 10  # Track last 10 steps for success rate

            for step in range(points_per_start - 1):
                if call_count >= budget or check_deadline():
                    break

                # Generate population
                improvements = 0
                for _ in range(pop_size):
                    if call_count >= budget or check_deadline():
                        break

                    y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                    fy = func(y)
                    call_count += 1

                    if fy < fx:
                        # Move towards y
                        x = y
                        fx = fy
                        improvements += 1
                        success_count += 1

                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()

                # Adapt sigma based on success rate
                if improvements > 0:
                    sigma *= 1.1
                else:
                    sigma *= 0.85

                # Clamp sigma
                sigma = np.clip(sigma, 0.01, 5.0)

                # Restart if stuck for too long
                if success_count == 0 and step > 50:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
