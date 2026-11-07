"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Restarting (1+1)-ES with 1/5 success rule, deadline guard, and final polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Global best tracking
        x_best = rng.uniform(-5, 5, dim)
        f_best = func(x_best)
        calls_made = 1

        # Parameters for restart strategy
        stagnation_threshold = 30

        # Main search loop with restarts
        while calls_made < budget:
            # Check time limit
            if time.time() > deadline:
                break

            # Initialize current run state
            x = x_best.copy()
            # Start with a moderate sigma
            sigma = 2.0
            stagnation_count = 0
            local_best_val = f_best
            local_best_x = x_best.copy()

            # Run (1+1)-ES for this restart phase
            while calls_made < budget:
                if time.time() > deadline:
                    break

                # Generate neighbor
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < local_best_val:
                    # Improvement
                    local_best_val = fy
                    local_best_x = y
                    x = y
                    sigma *= 1.5
                    stagnation_count = 0

                    # Update global best
                    if fy < f_best:
                        f_best = fy
                        x_best = y
                else:
                    # No improvement
                    sigma *= 0.87
                    stagnation_count += 1

                # Check for stagnation to trigger restart
                if stagnation_count >= stagnation_threshold:
                    break

            # If sigma becomes too small, restart
            if sigma < 1e-6 and calls_made < budget:
                break

            # If we've made no global progress in this entire restart cycle,
            # and we've used a significant chunk of budget, try a larger jump
            if local_best_val >= f_best and calls_made < budget * 0.8:
                # Do a larger random jump from x_best to escape potential local minima
                sigma_jump = 3.0
                y_jump = np.clip(x_best + sigma_jump * rng.standard_normal(dim), -5, 5)
                fy_jump = func(y_jump)
                calls_made += 1
                if fy_jump < f_best:
                    f_best = fy_jump
                    x_best = y_jump

        # Final high-precision polish pass
        remaining = budget - calls_made
        if remaining > 5 and time.time() < deadline:
            # Use a very small sigma for fine-tuning
            sigma_polish = 0.05
            x = x_best.copy()
            fx = f_best

            # Adaptive polish: shrink sigma as we get closer
            for i in range(min(remaining, 50)):
                if time.time() > deadline or calls_made >= budget:
                    break

                y = np.clip(x + sigma_polish * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma_polish *= 0.9  # Shrink step size on improvement
                    if fx < f_best:
                        f_best = fx
                        x_best = x
                else:
                    sigma_polish *= 0.95  # Shrink slightly even on rejection

                # If sigma becomes negligibly small, stop polishing
                if sigma_polish < 1e-8:
                    break

        return x_best, f_best
# EVOLVE-BLOCK-END
