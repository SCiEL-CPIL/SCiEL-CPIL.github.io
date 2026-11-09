"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with time guard, multi-start, and polish phase."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        def time_left():
            return deadline - time.time() > 5.0

        # Multi-start: try both known basin regions of Lunacek bi-Rastrigin
        # Basin 1: near origin, Basin 2: near the shifted point
        init_points = [
            np.zeros(dim),
            np.ones(dim) * 2.5,
            np.ones(dim) * -2.5,
            rng.uniform(-4, 4, dim),
        ]

        best_x = None
        best_f = np.inf
        calls = 0

        for init in init_points:
            if not time_left() or calls >= budget:
                break
            x = init.copy()
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Run (1+1)-ES from this start
            sigma = 1.5
            success_count = 0
            total_count = 0
            for _ in range(budget - calls - 1):
                if not time_left() or calls >= budget - 50:  # reserve for polish
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                total_count += 1
                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.15  # gentle increase on success
                else:
                    sigma *= 0.92  # gentle decrease on failure

                # Periodic restart if stuck
                if total_count % 50 == 0 and success_count / total_count < 0.1:
                    sigma = max(sigma * 2.0, 0.5)

        # Final polish phase: use remaining budget for high-precision convergence
        if best_x is not None and time_left() and calls < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            for _ in range(budget - calls):
                if not time_left() or calls >= budget:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma *= 1.1
                else:
                    sigma *= 0.95

                # Reduce sigma further as we converge
                if calls > budget * 0.9:
                    sigma *= 0.98

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
