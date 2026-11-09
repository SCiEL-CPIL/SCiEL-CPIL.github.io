"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES style optimizer with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        # Use a deterministic seed for reproducibility
        rng = np.random.default_rng(123)

        best_x = None
        best_f = np.inf

        # Number of restarts and budget per restart
        # We'll adapt based on time: if fast, do more restarts
        # Start with a moderate number of restarts
        n_restarts = 5
        budget_per_restart = max(1, (budget - n_restarts) // n_restarts)

        # For the first restart, use a larger budget to explore broadly
        # For subsequent restarts, focus near the best point found

        for restart in range(n_restarts):
            if not time_ok():
                break

            if restart == 0:
                # First restart: broad exploration
                x = rng.uniform(-5, 5, dim)
                sigma = 3.0
            else:
                # Subsequent restarts: start near best found, with some noise
                if best_x is not None:
                    noise_scale = 2.0 / (restart + 1)
                    x = np.clip(best_x + noise_scale * rng.standard_normal(dim), -5, 5)
                else:
                    x = rng.uniform(-5, 5, dim)
                sigma = 1.0 / (restart + 1)

            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # CMA-ES-like update with adaptive sigma
            # Use a simple (1+lambda)-ES with step size adaptation
            lambda_ = 4  # number of offspring per generation
            successes = 0
            total_steps = 0
            max_steps = budget_per_restart

            for step in range(max_steps):
                if not time_ok():
                    break

                # Generate lambda offspring
                offspring = []
                for j in range(lambda_):
                    if not time_ok():
                        break
                    y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    offspring.append((fy, y))

                if not offspring:
                    break

                # Sort by fitness
                offspring.sort(key=lambda t: t[0])
                f_new, y_new = offspring[0]

                if f_new < fx:
                    # Success
                    x = y_new
                    fx = f_new
                    successes += 1
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85

                total_steps += 1

                # Check if we should stop early if no improvement in a while
                # (optional: could add stagnation detection)

        return best_x, best_f
# EVOLVE-BLOCK-END
