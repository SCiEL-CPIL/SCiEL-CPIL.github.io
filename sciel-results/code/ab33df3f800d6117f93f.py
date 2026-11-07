"""Optimizer for BBOB-style f13 (Sharp ridge) with multi-start and adaptive (1+1)-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive step size for sharp ridge optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(42)

        # Phase 1: Global exploration - evaluate 50 random points
        n_init = 50
        points = []
        values = []
        for _ in range(n_init):
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            points.append(x)
            values.append(fx)

        # Select top 5 starting points
        values_arr = np.array(values)
        top_indices = np.argsort(values_arr)[:5]
        best_points = [points[i] for i in top_indices]
        best_values = [values[i] for i in top_indices]

        # Track global best
        global_best_x = best_points[0]
        global_best_f = best_values[0]
        calls_used = n_init

        # Phase 2: Run (1+1)-ES from each of the 5 starting points
        remaining_budget = self.budget - calls_used
        budget_per_start = max(1, remaining_budget // 5)

        for i in range(5):
            x = best_points[i].copy()
            fx = best_values[i]
            sigma = 1.5  # Initial step size

            # Adaptive (1+1)-ES with 1/5 success rule
            successes = 0
            total = 0
            window = 10

            for step in range(budget_per_start):
                # Check deadline
                if time.time() > deadline:
                    break
                if calls_used >= self.budget:
                    break

                # Generate offspring
                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_used += 1

                total += 1
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    sigma *= 1.25  # Increase step size on success
                    # Update global best
                    if fx < global_best_f:
                        global_best_f = fx
                        global_best_x = x.copy()
                else:
                    sigma *= 0.85  # Decrease step size on failure

                # Adapt sigma based on success rate in window
                if total % window == 0:
                    success_rate = successes / total
                    if success_rate > 0.4:
                        sigma *= 1.5
                    elif success_rate < 0.1:
                        sigma *= 0.5
                    successes = 0
                    total = 0

                # Stagnation detection: if no improvement for many steps, restart locally
                # (handled implicitly by the multi-start structure)

            # After local search, if this point is better, it's already updated globally

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
