"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit

        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1
        sigma = 1.0
        success = 0
        attempts = 0

        while calls < budget:
            if time.time() > deadline:
                break

            # 1/5 success rule: adjust sigma based on recent successes
            if attempts >= 5:
                rate = success / attempts
                if rate > 0.2:
                    sigma *= 1.5
                elif rate < 0.2:
                    sigma /= 1.5
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 10.0)
                success = 0
                attempts = 0

            # Generate trial step
            z = rng.standard_normal(dim)
            x_trial = x + sigma * z
            x_trial = np.clip(x_trial, -5, 5)
            f_trial = func(x_trial)
            calls += 1
            attempts += 1

            if f_trial < fx:
                success += 1
                x = x_trial.copy()
                fx = f_trial
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = x_trial.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
