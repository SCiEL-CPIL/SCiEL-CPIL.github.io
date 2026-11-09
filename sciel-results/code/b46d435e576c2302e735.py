"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with diverse initialization and adaptive sigma."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # 10s safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf

        # Use multiple restarts with diverse starting points
        # Each restart gets a portion of the budget
        num_restarts = 5
        budget_per_restart = max(1, (budget - 1) // num_restarts)

        # Diverse initialization points for Rastrigin landscape
        # Rastrigin has many local minima; we want to explore different regions
        init_points = []
        # Start near origin
        init_points.append(np.zeros(dim))
        # Random points spread across the domain
        for _ in range(num_restarts - 1):
            init_points.append(rng.uniform(-4, 4, dim))

        for restart_idx in range(num_restarts):
            if time.time() > deadline:
                break

            x = init_points[restart_idx].copy()
            fx = func(x)

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.5
            # Adaptive step size with success-rate adaptation
            successes = 0
            total_steps = 0
            window_size = 10

            for step in range(budget_per_restart):
                if time.time() > deadline:
                    break
                if step >= budget - 1:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                total_steps += 1

                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    sigma *= 1.3
                else:
                    sigma *= 0.85

                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

                # Adapt sigma based on success rate
                if total_steps % window_size == 0:
                    success_rate = successes / window_size
                    if success_rate > 0.5:
                        sigma *= 1.2
                    elif success_rate < 0.2:
                        sigma *= 0.7
                    successes = 0

                # Prevent sigma from becoming too small or too large
                sigma = np.clip(sigma, 0.01, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
