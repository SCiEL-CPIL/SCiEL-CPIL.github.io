"""Rewritten optimizer for BBOB f4 (Büche-Rastrigin) with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, multi-restart, and coordinate descent polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._deadline_margin = 10.0  # seconds before hard limit to stop

    def __call__(self, func):
        t_start = time.time()
        hard_limit = 300.0 - self._deadline_margin
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_ok():
            return (time.time() - t_start) < hard_limit

        def try_update(x, f):
            nonlocal best_x, best_f, calls_made
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Phase 1: Multi-start (1+1)-ES with adaptive sigma
        # Use several restarts to explore different basins
        num_restarts = 5
        calls_per_restart = max(1, (budget - num_restarts) // num_restarts)

        for r in range(num_restarts):
            if not time_ok() or calls_made >= budget:
                break

            # Initialize near previous best or random
            if best_x is not None and r > 0:
                # Perturb around best found so far
                x = best_x + rng.normal(0, 1.0, dim)
                x = np.clip(x, -5, 5)
            else:
                x = rng.uniform(-5, 5, dim)

            fx = func(x)
            calls_made += 1
            try_update(x, fx)

            sigma = 2.0
            # 1/5 success rule tracking
            successes = 0
            window = 0

            for _ in range(calls_per_restart):
                if not time_ok() or calls_made >= budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    try_update(x, fx)
                    successes += 1
                window += 1

                if window >= 10:
                    # 1/5 success rule
                    if successes / window > 0.2:
                        sigma *= 1.25
                    elif successes / window < 0.1:
                        sigma *= 0.8
                    successes = 0
                    window = 0

                # Clamp sigma
                sigma = np.clip(sigma, 1e-6, 10.0)

        # Phase 2: Coordinate descent polish on best point
        if best_x is not None and time_ok() and calls_made < budget:
            x = best_x.copy()
            fx = best_f
            step = 0.5

            # Multi-scale coordinate descent
            for scale in [1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01]:
                if not time_ok() or calls_made >= budget:
                    break
                step = scale
                for _ in range(3):  # Passes per scale
                    if not time_ok() or calls_made >= budget:
                        break
                    improved = False
                    for i in range(dim):
                        if not time_ok() or calls_made >= budget:
                            break
                        for direction in [1, -1]:
                            y = x.copy()
                            y[i] += direction * step
                            y = np.clip(y, -5, 5)
                            fy = func(y)
                            calls_made += 1
                            if fy < fx:
                                x, fx = y, fy
                                try_update(x, fx)
                                improved = True
                                break  # Move in this direction, continue to next coordinate
                    if not improved:
                        break

        # Phase 3: Fine-grained random search around best
        if best_x is not None and time_ok() and calls_made < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            remaining = budget - calls_made
            for _ in range(remaining):
                if not time_ok():
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1
                if fy < fx:
                    x, fx = y, fy
                    try_update(x, fx)
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                sigma = np.clip(sigma, 1e-6, 1.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
