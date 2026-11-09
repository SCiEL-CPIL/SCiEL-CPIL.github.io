"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """(1+1)-ES with aggressive exploration and time guards."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        # Time management
        start_time = time.time()
        time_limit = 325.0  # Safety margin before 330s kill

        # Initial State
        x = np.random.uniform(-5, 5, n)
        fx = func(x)
        x_best = x.copy()
        f_best = fx

        # Step size
        sigma = 3.0

        eval_count = 1

        # Track success rate for adaptive step size
        successes = 0
        total = 0
        window_size = 20

        while eval_count < budget:
            # Check time
            if time.time() - start_time > time_limit:
                break

            try:
                z = np.random.standard_normal(n)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
            except Exception:
                break

            eval_count += 1
            total += 1

            if fy < fx:
                # Success
                x = y
                fx = fy
                x_best = y.copy()
                f_best = fy
                successes += 1
                sigma *= 1.3
            else:
                # Failure
                sigma *= 0.85

            # Prevent sigma from exploding or becoming too small
            if sigma > 50:
                sigma = 50
            if sigma < 1e-14:
                break

            # Periodically adapt sigma based on success rate
            if total >= window_size:
                rate = successes / total
                if rate > 0.3:
                    sigma *= 1.2
                elif rate < 0.1:
                    sigma *= 0.8
                successes = 0
                total = 0

            # Occasionally try a larger random jump to escape local minima
            # Ensure we don't exceed the budget
            if eval_count % 50 == 0 and sigma < 1.0 and eval_count < budget - 1:
                jump = np.random.uniform(-3, 3, n)
                y_jump = x + jump
                y_jump = np.clip(y_jump, -5, 5)
                fy_jump = func(y_jump)
                eval_count += 1
                if fy_jump < fx:
                    x = y_jump
                    fx = fy_jump
                    x_best = y_jump.copy()
                    f_best = fy_jump
                    sigma = max(sigma, 2.0)

        return x_best, f_best
# EVOLVE-BLOCK-END
