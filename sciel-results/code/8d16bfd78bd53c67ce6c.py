"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with adaptive restarts, adaptive per-restart budget, and best-point re-seeding."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 275  # safety margin before 300s limit

        best_x = None
        best_f = np.inf
        calls_made = 0
        budget = self.budget
        dim = self.dim

        max_restarts = 15
        base_calls = max(1, budget // max_restarts)
        stagnation_threshold = 20

        for restart in range(max_restarts):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break

            # Adaptive calls per restart: explore cheaply early, exploit later
            if restart < 3:
                calls_this_restart = max(10, base_calls)
            elif restart < 8:
                calls_this_restart = max(20, base_calls * 2)
            else:
                calls_this_restart = max(30, base_calls * 3)
            calls_this_restart = min(calls_this_restart, budget - calls_made)
            if calls_this_restart <= 0:
                break

            # Initialization: center-biased early, exploit best point later, explore otherwise
            if best_x is not None and restart >= 5:
                # Re-seed near the best point found so far (exploit)
                x = np.clip(best_x + 1.5 * rng.standard_normal(dim), -5, 5)
            elif restart < 3:
                x = rng.uniform(-3, 3, dim)
            elif restart < 8:
                x = rng.uniform(-4, 4, dim)
            else:
                x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # (1+1)-ES with adaptive sigma
            sigma = 1.5
            sigma_min = 0.005
            sigma_max = 12.0
            no_improve = 0

            for step in range(calls_this_restart):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x, fx = y, fy
                    no_improve = 0
                    sigma = min(sigma * 1.12, sigma_max)
                else:
                    no_improve += 1
                    sigma = max(sigma * 0.92, sigma_min)

                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

                if no_improve > stagnation_threshold:
                    break

            # Early stop if deep in the basin
            if best_f < -80:
                break
            if best_f < -40 and restart >= 6:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
