"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with windowed 1/5 success rule for Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        # Start at origin: BBOB shifts keep optimum in [-4,4]^5
        x = np.zeros(dim)
        fx = func(x)
        calls = 1

        # Windowed 1/5 success rule parameters
        window = 15  # window size for success counting
        successes = 0
        sigma = 2.0  # initial step size
        min_sigma = 1e-8
        max_sigma = 10.0

        best_x = x.copy()
        best_f = fx

        for _ in range(budget - 1):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            # Generate one offspring
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1

            # Track global best
            if fy < best_f:
                best_f = fy
                best_x = y.copy()

            # Update current best
            if fy < fx:
                x = y
                fx = fy
                successes += 1
            else:
                # Keep x, don't update
                pass

            # Windowed 1/5 success rule: adapt sigma every 'window' steps
            if calls % window == 0:
                # 1/5 rule: increase if >1/5 success, decrease if <1/5, else keep
                success_rate = successes / window
                if success_rate > 0.2:
                    sigma *= 1.3  # increase step size
                elif success_rate < 0.2:
                    sigma *= 0.7  # decrease step size
                # else: keep sigma unchanged
                sigma = np.clip(sigma, min_sigma, max_sigma)
                successes = 0  # reset counter for next window

        return best_x, best_f
# EVOLVE-BLOCK-END
