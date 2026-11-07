"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derivative-free optimizer using adaptive (1+1)-ES with restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 330s kill

        rng = np.random.default_rng(7)

        # Best solution found overall
        best_x = np.zeros(dim)
        best_f = func(best_x)
        call_count = 1

        # Adaptive (1+1)-ES with success rate control
        # Target success rate ~0.33
        x = best_x.copy()
        fx = best_f
        sigma = 1.0

        # Parameters for sigma adaptation
        # Use a simple adaptation: if success rate in last N trials > target, increase sigma
        # Otherwise decrease
        success_window = 15
        successes = []
        sigma_increase = 1.1
        sigma_decrease = 0.9

        # Stagnation detection
        stagnation_threshold = 50
        stagnation_count = 0
        last_best_f = best_f

        for _ in range(budget - 1):
            if time.time() > deadline:
                break

            # Generate trial
            y = x + sigma * rng.standard_normal(dim)
            y = np.clip(y, -5, 5)
            fy = func(y)
            call_count += 1

            success = fy < fx
            successes.append(1 if success else 0)
            if len(successes) > success_window:
                successes.pop(0)

            if success:
                x = y
                fx = fy
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                    stagnation_count = 0
                else:
                    stagnation_count += 1
            else:
                stagnation_count += 1

            # Adapt sigma based on success rate
            if len(successes) == success_window:
                rate = np.mean(successes)
                if rate > 0.4:
                    sigma *= sigma_increase
                elif rate < 0.25:
                    sigma *= sigma_decrease

            # Clamp sigma
            sigma = np.clip(sigma, 0.01, 10.0)

            # Restart if stagnant
            if stagnation_count >= stagnation_threshold:
                # Try a few random restarts
                for _ in range(3):
                    if time.time() > deadline:
                        break
                    x_new = rng.uniform(-5, 5, dim)
                    fx_new = func(x_new)
                    call_count += 1
                    if fx_new < best_f:
                        best_f = fx_new
                        best_x = x_new.copy()
                        x = x_new
                        fx = fx_new
                        sigma = 1.0
                        stagnation_count = 0
                        successes = []
                        break
                else:
                    # No improvement, reduce sigma and continue
                    sigma = max(sigma * 0.5, 0.01)
                    stagnation_count = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
