"""Simulated Annealing with adaptive cooling for BBOB f7 (Step ellipsoidal)."""
import numpy as np
import time


class Optimizer:
    """Simulated Annealing with adaptive cooling schedule."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(12345)
        n = self.dim
        budget = self.budget

        # Initial broad scout
        x = rng.uniform(-5, 5, n)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Adaptive step size
        step_size = 3.0
        min_step = 1e-6
        max_step = 8.0

        # Temperature schedule: start high, cool geometrically
        T = 50.0
        T_min = 1e-10
        cooling_rate = 0.995

        # Track for adaptive step adjustment
        success_count = 0
        total_count = 0
        window_size = 50

        calls_made = 1

        for i in range(budget - 1):
            # Deadline guard
            if time.time() > deadline:
                break

            # Generate candidate
            perturbation = rng.standard_normal(n) * step_size
            y = np.clip(x + perturbation, -5, 5)
            fy = func(y)
            calls_made += 1

            total_count += 1
            delta_f = fy - fx

            # Acceptance criterion
            if delta_f < 0:
                # Always accept improvements
                x, fx = y, fy
                success_count += 1
                # Update best
                if fx < best_fx:
                    best_x = x.copy()
                    best_fx = fx
                    # Increase step size slightly on improvement
                    step_size = min(max_step, step_size * 1.1)
            else:
                # Metropolis acceptance: uphill moves with probability exp(-delta/T)
                if T > T_min:
                    prob = np.exp(-delta_f / T)
                    if rng.random() < prob:
                        x, fx = y, fy
                        # Don't count as success for step adaptation
                # else: only accept improvements (T too low)

            # Adaptive cooling
            T *= cooling_rate

            # Adaptive step size adjustment every window_size evaluations
            if total_count % window_size == 0:
                success_rate = success_count / window_size
                if success_rate > 0.6:
                    step_size = min(max_step, step_size * 1.3)
                elif success_rate < 0.3:
                    step_size = max(min_step, step_size * 0.7)
                success_count = 0
                total_count = 0

            # If step size is very small, try a larger random jump to escape plateaus
            if step_size < 0.01 and i % 100 == 0:
                jump = rng.uniform(-5, 5, n)
                y_jump = np.clip(x + jump, -5, 5)
                fy_jump = func(y_jump)
                calls_made += 1
                if fy_jump < fx:
                    x, fx = y_jump, fy_jump
                    step_size = 2.0
                    if fx < best_fx:
                        best_x = x.copy()
                        best_fx = fx

        return best_x, best_fx
