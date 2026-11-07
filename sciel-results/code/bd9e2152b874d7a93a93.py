"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid (1+1)-ES with adaptive step size, restarts, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_fx = func(best_x)
        calls_made = 1

        # (1+1)-ES parameters
        sigma = 2.0
        sigma_min = 0.01
        sigma_max = 5.0
        success_rate = 0.0
        target_success_rate = 1.0 / (dim + 1)
        adaptation_window = dim + 1

        # Restart strategy
        max_no_improve = 80
        no_improve_count = 0
        restart_count = 0

        while calls_made < budget:
            if time.time() > deadline:
                break

            # Generate candidate
            step = sigma * rng.standard_normal(dim)
            candidate = best_x + step
            candidate = np.clip(candidate, -5, 5)

            f_candidate = func(candidate)
            calls_made += 1

            if f_candidate < best_fx:
                best_x = candidate.copy()
                best_fx = f_candidate
                success_rate = 1.0
                no_improve_count = 0
            else:
                success_rate = 0.0
                no_improve_count += 1

            # Adapt step size based on success rate over a window
            # Simple adaptive rule: increase sigma on success, decrease on failure
            if f_candidate < best_fx:
                sigma *= 1.1
            else:
                sigma *= 0.9

            sigma = np.clip(sigma, sigma_min, sigma_max)

            # Check for restart
            if no_improve_count >= max_no_improve:
                restart_count += 1
                # Restart: jump to a new random point, keep global best
                # Use a large jump from the current best
                jump = 3.0 * rng.standard_normal(dim)
                new_point = np.clip(best_x + jump, -5, 5)
                f_new = func(new_point)
                calls_made += 1

                if f_new < best_fx:
                    best_x = new_point.copy()
                    best_fx = f_new

                # Reset local search
                sigma = 2.0
                no_improve_count = 0

                # If we've restarted many times, try a completely random point
                if restart_count % 3 == 0 and calls_made < budget:
                    random_point = rng.uniform(-5, 5, dim)
                    f_random = func(random_point)
                    calls_made += 1
                    if f_random < best_fx:
                        best_x = random_point.copy()
                        best_fx = f_random
                    sigma = 3.0

        return best_x, best_fx
# EVOLVE-BLOCK-END
