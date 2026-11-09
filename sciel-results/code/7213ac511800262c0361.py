"""Step ellipsoidal (f7) optimizer: plateau-aware (1+1)-ES with restarts and local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with plateau handling, random restarts, and a final local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 300s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls = 0

        def remaining_budget():
            return budget - calls

        def check_deadline():
            return time.time() < deadline

        # Phase 1: (1+1)-ES with plateau-aware sigma adaptation and restarts
        max_restarts = 8
        plateau_threshold = 30  # no improvement after this many steps -> consider restart
        max_sigma = 10.0
        min_sigma = 1e-6

        for restart in range(max_restarts):
            if not check_deadline() or calls >= budget:
                break

            # Initialize: first restart from center, others random
            if restart == 0:
                x = np.zeros(dim)
            else:
                x = rng.uniform(-4.5, 4.5, dim)
            fx = func(x)
            calls += 1

            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 2.0
            no_improve = 0
            steps_in_restart = 0
            max_steps_per_restart = 150

            while calls < budget and steps_in_restart < max_steps_per_restart:
                if not check_deadline():
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                steps_in_restart += 1

                if fy < fx:
                    # Improvement
                    x, fx = y, fy
                    no_improve = 0
                    # Adapt sigma: increase on success
                    sigma = min(sigma * 1.4, max_sigma)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    no_improve += 1
                    # Decrease sigma on failure
                    sigma = max(sigma * 0.85, min_sigma)

                    # Plateau detection: if stuck for too long, try a larger jump
                    if no_improve >= plateau_threshold:
                        # Try a large random step to escape plateau
                        jump_sigma = sigma * 5.0
                        y_jump = np.clip(x + jump_sigma * rng.standard_normal(dim), -5, 5)
                        fy_jump = func(y_jump)
                        calls += 1
                        steps_in_restart += 1

                        if fy_jump < fx:
                            x, fx = y_jump, fy_jump
                            no_improve = 0
                            sigma = min(jump_sigma * 0.5, max_sigma)
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                        else:
                            # Reset to a new random point if still stuck
                            if no_improve >= 60:
                                break
                            # Try a moderate perturbation
                            y_mod = np.clip(x + sigma * 2.0 * rng.standard_normal(dim), -5, 5)
                            fy_mod = func(y_mod)
                            calls += 1
                            steps_in_restart += 1
                            if fy_mod < fx:
                                x, fx = y_mod, fy_mod
                                no_improve = 0
                                sigma = min(sigma * 1.2, max_sigma)
                                if fx < best_f:
                                    best_f = fx
                                    best_x = x.copy()
                            else:
                                no_improve += 1

            # If we broke out due to plateau, continue to next restart

        # Phase 2: Local polish around the best point found
        if best_x is not None and check_deadline() and calls < budget:
            # Use a smaller sigma for fine-tuning
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            polish_steps = min(50, budget - calls)
            no_improve = 0

            for _ in range(polish_steps):
                if not check_deadline() or calls >= budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1

                if fy < fx:
                    x, fx = y, fy
                    no_improve = 0
                    sigma = min(sigma * 1.3, 0.5)
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    no_improve += 1
                    sigma = max(sigma * 0.8, 1e-5)
                    if no_improve > 20:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
