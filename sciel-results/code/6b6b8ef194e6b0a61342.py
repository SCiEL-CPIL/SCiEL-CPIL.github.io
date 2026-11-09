"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Simulated Annealing with adaptive cooling for BBOB f7 (Step ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Initialize from broad uniform scout
        x = rng.uniform(-5, 5, n)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Adaptive step size parameters
        sigma = 2.0
        min_sigma = 0.01
        max_sigma = 8.0

        # Temperature schedule
        T = 10.0
        cooling = 0.995

        # Track stagnation for adaptive behavior
        stagnation_count = 0
        last_improvement = 0
        eval_count = 0

        while eval_count < budget - 1:
            # Check time limit
            if time.time() > deadline:
                break

            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)
            eval_count += 1

            # Acceptance criterion
            if fy < fx:
                # Always accept improvements
                x, fx = y, fy
                if fy < best_fx:
                    best_x = x.copy()
                    best_fx = fy
                    stagnation_count = 0
                    last_improvement = eval_count
                    # Reduce temperature on improvement
                    T *= 0.95
                else:
                    stagnation_count += 1
                # Adjust sigma based on improvement
                sigma = min(max(sigma * 1.1, min_sigma), max_sigma)
            else:
                # Metropolis acceptance for uphill moves
                delta_f = fy - fx
                if delta_f < 0:
                    accept = True
                elif T > 1e-10:
                    accept = rng.random() < np.exp(-delta_f / T)
                else:
                    accept = False

                if accept:
                    x = y
                    fx = fy
                    stagnation_count += 1
                else:
                    stagnation_count += 1

                # Cool down
                T *= cooling

                # Adaptive sigma: increase if stuck, decrease if moving
                if stagnation_count > 20:
                    sigma = min(sigma * 1.3, max_sigma)
                    T = max(T, 1.0)  # Reheat to escape plateau
                    stagnation_count = 0
                elif stagnation_count > 5:
                    sigma = min(sigma * 1.1, max_sigma)
                else:
                    sigma = max(sigma * 0.95, min_sigma)

        return best_x, best_fx
# EVOLVE-BLOCK-END
