"""Optimizer for BBOB f7 (Step ellipsoidal) with multi-start and adaptive ES."""
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with multi-start, adaptive step size, and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 285.0  # Safety margin before 300s limit
        d = self.dim
        rng = np.random.default_rng(42)

        def time_ok():
            return (time.time() - start_time) < timeout

        # Multi-start initialization: evaluate several random points, keep best
        n_starts = 15
        best_x = None
        best_fx = np.inf

        for _ in range(n_starts):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, d)
            fx = func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(d)
            best_fx = func(best_x)

        # Now run (1+1)-ES from the best starting point
        x = best_x.copy()
        fx = best_fx
        sigma = 1.5
        calls_made = n_starts  # approximate; we don't track exactly but stay under budget

        # Track remaining budget
        remaining = self.budget - calls_made

        # Adaptive step size parameters
        sigma_min = 1e-8
        sigma_max = 10.0
        # 1/5 success rule adaptation
        success_count = 0
        window_size = 10
        window = []

        for i in range(remaining):
            if not time_ok():
                break

            # Generate candidate
            step = sigma * rng.standard_normal(d)
            y = np.clip(x + step, -5, 5)
            fy = func(y)

            if fy < fx:
                x = y
                fx = fy
                success_count += 1
                window.append(1)
                # Increase sigma on success
                sigma = min(sigma * 1.3, sigma_max)
            else:
                window.append(0)
                # Decrease sigma on failure
                sigma = max(sigma * 0.8, sigma_min)

            # Periodically adapt sigma based on success rate over window
            if len(window) >= window_size:
                success_rate = sum(window) / len(window)
                # Target success rate ~0.2 (1/5 rule)
                if success_rate > 0.3:
                    sigma = min(sigma * 1.5, sigma_max)
                elif success_rate < 0.1:
                    sigma = max(sigma * 0.6, sigma_min)
                window = []

            # Occasionally try a larger exploratory step
            if i % 50 == 0 and i > 0:
                # Try a few random perturbations to escape local minima
                for _ in range(3):
                    if not time_ok():
                        break
                    big_step = 2.0 * rng.standard_normal(d)
                    y2 = np.clip(x + big_step, -5, 5)
                    fy2 = func(y2)
                    if fy2 < fx:
                        x = y2
                        fx = fy2

        return x, fx
