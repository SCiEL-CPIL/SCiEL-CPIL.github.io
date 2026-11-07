"""Optimizer for BBOB-style f17 (Schaffers F7) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Two-phase (1+1)-ES with:
      Phase 1: Coarse exploration from center with adaptive sigma and
               periodic global restarts to escape local minima.
      Phase 2: Fine-grained local refinement around the best point found.
    Designed for early progress (AOCC) on Schaffers F7 in 5-D.
    """

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Start at center (minimizer known to be in [-4,4]^5)
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls_made = 1

        # --- Phase 1: Coarse search with adaptive sigma and restarts ---
        sigma = 1.5
        sigma_min = 1e-5
        sigma_max = 8.0
        sigma_up = 1.3
        sigma_down = 0.85

        # Restart mechanism: if no improvement for `stuck_threshold` steps,
        # jump to a new random point (but keep best_x/best_fx).
        stuck_counter = 0
        stuck_threshold = 40
        max_restarts = 15
        restarts_used = 0

        while calls_made < budget:
            if time.time() > deadline:
                break

            # Check for stagnation -> global restart
            if stuck_counter >= stuck_threshold and restarts_used < max_restarts:
                # Jump to a new random location
                x_new = rng.uniform(-4.5, 4.5, dim)
                fx_new = func(x_new)
                calls_made += 1
                restarts_used += 1
                stuck_counter = 0
                if fx_new < best_fx:
                    best_fx = fx_new
                    best_x = x_new.copy()
                # Move current search to the new point
                x = x_new
                fx = fx_new
                sigma = 1.5  # reset sigma for exploration
                if calls_made >= budget:
                    break
                continue

            # Generate candidate
            step = sigma * rng.standard_normal(dim)
            y = np.clip(x + step, -5, 5)
            fy = func(y)
            calls_made += 1

            if fy < fx:
                x = y
                fx = fy
                stuck_counter = 0
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                sigma = min(sigma * sigma_up, sigma_max)
            else:
                sigma = max(sigma * sigma_down, sigma_min)
                stuck_counter += 1

        # --- Phase 2: Fine-grained local refinement around best point ---
        if calls_made < budget:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.3  # start with a moderate step for refinement
            sigma_min_p2 = 1e-7
            sigma_max_p2 = 1.0
            sigma_up_p2 = 1.1
            sigma_down_p2 = 0.92

            while calls_made < budget:
                if time.time() > deadline:
                    break

                step = sigma * rng.standard_normal(dim)
                y = np.clip(x + step, -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x = y
                    fx = fy
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                    sigma = min(sigma * sigma_up_p2, sigma_max_p2)
                else:
                    sigma = max(sigma * sigma_down_p2, sigma_min_p2)
                    if sigma <= sigma_min_p2 * 1.01:
                        break

        return best_x, best_fx
# EVOLVE-BLOCK-END
