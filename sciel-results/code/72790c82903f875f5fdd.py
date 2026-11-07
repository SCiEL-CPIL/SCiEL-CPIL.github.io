"""Optimized optimizer for BBOB f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+8)-ES with adaptive step size and deadline guard for f13 Sharp Ridge."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        deadline = time.time() + 280.0  # safety margin before 330s kill

        # Initialize at center (0,0,0,0,0) which is a reasonable starting point
        # for a rotated/shifted landscape in [-5,5]^5
        x = np.zeros(self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # (1+8)-ES parameters
        sigma = 1.5
        n_offspring = 8
        target_success = 0.2  # 20% success rate target
        alpha = 0.5  # adaptation speed

        calls_made = 1

        while calls_made < self.budget:
            if time.time() > deadline:
                break

            # Generate offspring
            best_offspring = None
            best_offspring_fx = np.inf
            successes = 0

            for _ in range(n_offspring):
                if time.time() > deadline:
                    break
                if calls_made >= self.budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    successes += 1
                if fy < best_offspring_fx:
                    best_offspring_fx = fy
                    best_offspring = y

            # Update step size based on success rate
            if successes > 0:
                success_rate = successes / n_offspring
                # Adapt sigma to target success rate
                if success_rate > target_success:
                    sigma *= (1 + alpha * (success_rate / target_success - 1))
                else:
                    sigma *= (1 - alpha * (1 - success_rate / target_success))

            # Clamp sigma to reasonable bounds
            sigma = np.clip(sigma, 0.01, 10.0)

            # Accept best offspring if it improves
            if best_offspring is not None and best_offspring_fx < fx:
                x = best_offspring
                fx = best_offspring_fx

            # Track global best
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

            # If we've made good progress and sigma is very small,
            # we can stop early to save time
            if sigma < 0.02 and calls_made > self.budget * 0.5:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
