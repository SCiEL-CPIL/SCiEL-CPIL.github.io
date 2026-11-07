"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start (1+1)-ES with adaptive sigma, restarts, and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        deadline = time.time() + 320  # 10s safety margin before 330s limit

        best_x = None
        best_f = np.inf
        calls_made = 0

        def try_call(x):
            nonlocal calls_made
            if calls_made >= budget:
                return None
            fx = func(np.clip(x, -5, 5))
            calls_made += 1
            return fx

        # Multi-start with diverse initializations
        n_starts = 12
        points_per_start = max(1, (budget - n_starts) // n_starts)

        for start_i in range(n_starts):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break

            # Diverse starting points: mix of random, center, and structured
            if start_i == 0:
                x = np.zeros(dim)
            elif start_i == 1:
                x = rng.uniform(-3, 3, dim)
            elif start_i == 2:
                x = rng.uniform(-5, 5, dim)
            elif start_i == 3:
                # Structured: alternating signs
                x = rng.choice([-3, -1.5, 0, 1.5, 3], dim)
            elif start_i == 4:
                # Gaussian centered at 0
                x = rng.normal(0, 2, dim)
            elif start_i == 5:
                # Another random
                x = rng.uniform(-4, 4, dim)
            else:
                x = rng.uniform(-5, 5, dim)

            fx = try_call(x)
            if fx is None:
                break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # (1+1)-ES with adaptive sigma for this start
            sigma = 1.5
            stagnation = 0
            local_best = fx
            local_best_x = x.copy()

            for _ in range(points_per_start):
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break

                # Adaptive sigma: increase on success, decrease on failure
                # Anti-stagnation: kick if no improvement for a while
                if stagnation > 8:
                    # Large perturbation to escape local minimum
                    sigma = max(sigma, 2.0)
                    perturbation = sigma * 1.5 * rng.standard_normal(dim)
                else:
                    perturbation = sigma * rng.standard_normal(dim)

                y = np.clip(local_best_x + perturbation, -5, 5)
                fy = try_call(y)
                if fy is None:
                    break

                if fy < local_best - 1e-12:
                    local_best = fy
                    local_best_x = y.copy()
                    sigma *= 1.3
                    stagnation = 0
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                else:
                    sigma *= 0.85
                    stagnation += 1

                # Cap sigma
                sigma = np.clip(sigma, 0.01, 5.0)

        # Final polish: fine-tune around the best point found
        if best_x is not None and calls_made < budget:
            sigma = 0.3
            for _ in range(20):
                if time.time() > deadline:
                    break
                if calls_made >= budget:
                    break
                y = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                fy = try_call(y)
                if fy is None:
                    break
                if fy < best_f - 1e-12:
                    best_f = fy
                    best_x = y.copy()
                    sigma *= 1.2
                else:
                    sigma *= 0.8
                sigma = np.clip(sigma, 0.005, 1.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
