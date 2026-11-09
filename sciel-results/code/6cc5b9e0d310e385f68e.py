"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+lambda)-ES with adaptive step size, time guard, and restarts for Katsuura f23."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        d = self.dim
        total_budget = self.budget
        calls_made = 0
        best_x = None
        best_f = np.inf

        # Use a fixed seed for reproducibility
        rng = np.random.default_rng(42)

        # Strategy: Restarting (1+lambda)-ES
        # Katsuura is multimodal. We use a larger population and restart if stagnation occurs.
        lambda_ = 10
        initial_sigma = 2.0
        min_sigma = 1e-6
        max_sigma = 8.0

        # State for current run
        x = rng.uniform(-4, 4, d)
        fx = func(x)
        calls_made += 1
        best_x = x.copy()
        best_f = fx

        sigma = initial_sigma
        successes = 0
        attempts = 0
        stagnation_counter = 0
        max_stagnation = 60  # Restart if no global improvement in 60 calls

        while calls_made < total_budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate lambda_ offspring
            offspring = []
            offspring_f = []

            for i in range(lambda_):
                if calls_made >= total_budget:
                    break
                if time.time() > deadline:
                    break

                # Adaptive sigma adjustment every 10 attempts
                if attempts > 0 and attempts % 10 == 0:
                    rate = successes / attempts
                    if rate > 0.2:
                        sigma *= 1.2
                    elif rate < 0.2:
                        sigma *= 0.85
                    successes = 0
                    attempts = 0

                sigma = np.clip(sigma, min_sigma, max_sigma)

                y = np.clip(x + sigma * rng.standard_normal(d), -5, 5)
                fy = func(y)
                calls_made += 1
                attempts += 1

                offspring.append(y)
                offspring_f.append(fy)

                if fy < fx:
                    successes += 1

                # Track global stagnation
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1

            # Select best offspring to be the new parent
            if offspring:
                best_idx = np.argmin(offspring_f)
                if offspring_f[best_idx] < fx:
                    x = offspring[best_idx].copy()
                    fx = offspring_f[best_idx]

            # Check for restart
            if stagnation_counter > max_stagnation and calls_made < total_budget - 50:
                # Restart with a new random point
                x = rng.uniform(-4, 4, d)
                fx = func(x)
                calls_made += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                    stagnation_counter = 0
                else:
                    stagnation_counter = 0 # Reset counter for new run
                sigma = initial_sigma
                successes = 0
                attempts = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
