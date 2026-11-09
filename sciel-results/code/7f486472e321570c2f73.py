"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Adaptive (1+1)-ES with 1/5 success rule for BBOB Sphere."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin before 300s limit

        # Fixed seed for reproducibility
        rng = np.random.default_rng(42)

        # Start at the center (0) to minimize initial distance to hidden minimum
        x = np.zeros(dim)
        fx = func(x)
        calls = 1
        best_x = x.copy()
        best_f = fx

        # Initial step size - moderate to explore quickly
        sigma = 1.0

        # 1/5 success rule parameters
        window_size = 10
        successes = []

        while calls < budget and time.time() < deadline:
            # Generate perturbation
            dx = rng.standard_normal(dim)
            x_new = x + sigma * dx
            x_new = np.clip(x_new, -5, 5)

            fx_new = func(x_new)
            calls += 1

            is_success = fx_new < fx

            if is_success:
                x = x_new
                fx = fx_new
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                successes.append(1)
            else:
                successes.append(0)

            # Trim window
            if len(successes) > window_size:
                successes.pop(0)

            # Adapt sigma using 1/5 success rule
            if len(successes) == window_size:
                success_rate = sum(successes) / window_size
                # Target success rate is 1/5 = 0.2
                if success_rate > 0.2:
                    sigma *= 1.1
                elif success_rate < 0.2:
                    sigma *= 0.9
                else:
                    sigma *= 1.0

                # Clamp sigma to reasonable bounds
                sigma = np.clip(sigma, 1e-8, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
