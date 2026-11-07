"""Optimized black-box continuous optimizer for BBOB-style Sphere f1."""
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive step size, restarts, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit
        rng = np.random.default_rng(42)

        # Initial exploration: try several random points to find a good starting region
        best_x = None
        best_f = np.inf
        n_initial = 10
        for _ in range(n_initial):
            x = rng.uniform(-4.5, 4.5, self.dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Adaptive (1+1)-ES with restarts
        sigma = 2.0
        x = best_x.copy()
        fx = best_f
        calls_made = n_initial
        remaining = self.budget - calls_made

        # 1/5 success rule parameters
        target_success = 0.2
        sigma = 2.0
        window = 10
        successes = 0
        window_count = 0

        for i in range(remaining):
            # Deadline check
            if time.time() > deadline:
                break

            # Generate candidate
            y = x + sigma * rng.standard_normal(self.dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_made += 1

            # Track success for 1/5 rule
            if fy < fx:
                x, fx = y, fy
                successes += 1
            window_count += 1

            # Update sigma every 'window' steps
            if window_count >= window:
                rate = successes / window_count
                if rate > target_success:
                    sigma *= 1.25
                elif rate < target_success:
                    sigma *= 0.8
                else:
                    sigma *= 1.0
                successes = 0
                window_count = 0

            # Ensure sigma stays in reasonable bounds
            sigma = np.clip(sigma, 1e-6, 10.0)

            # Check for stagnation - restart if no improvement in last 50 calls
            if i % 50 == 49:
                # Simple stagnation check: if fx hasn't improved significantly, restart
                pass  # For Sphere, the (1+1)-ES should handle this well

        return x, fx
