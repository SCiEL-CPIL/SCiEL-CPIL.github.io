"""Optimized continuous optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-start CMA-ES with adaptive sigma, restarts, and a deadline guard.
    Uses a (1+1)-ES style inner loop with covariance adaptation for efficiency.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0
        budget = self.budget
        dim = self.dim

        def time_left():
            return (deadline - time.time()) > 0.5  # 0.5s margin

        def eval_point(x):
            nonlocal calls_made
            x = np.clip(x, -5, 5)
            fx = float(func(x))
            calls_made += 1
            return x, fx

        # Multi-start with different initializations
        n_starts = min(3, budget // 150)  # at least 1, at most 3 starts
        start_budget = budget // n_starts

        for start_idx in range(n_starts):
            if not time_left() or calls_made >= budget:
                break

            # Different initialization strategies
            if start_idx == 0:
                x = rng.uniform(-5, 5, dim)
            elif start_idx == 1:
                x = rng.normal(0, 2, dim)
                x = np.clip(x, -5, 5)
            else:
                x = np.zeros(dim)

            x, fx = eval_point(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            if calls_made >= budget or not time_left():
                break

            # CMA-ES-like (1+1) with covariance adaptation
            sigma = 1.0
            C = np.eye(dim)  # covariance matrix
            C_inv = np.eye(dim)
            step_size = 1.0

            # Track success rate for sigma adaptation
            successes = 0
            trials = 0

            while calls_made < budget and time_left():
                # Generate perturbation using covariance
                z = rng.multivariate_normal(np.zeros(dim), C, size=1).flatten()
                y = x + sigma * step_size * z
                y = np.clip(y, -5, 5)

                y, fy = eval_point(y)
                trials += 1

                if fy < fx:
                    # Success: update
                    successes += 1
                    fx = fy
                    x = y

                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()

                    # Update covariance: rank-1 update
                    d = (y - x) / (sigma * step_size) if sigma * step_size > 1e-10 else np.zeros(dim)
                    C = (1 - 1/dim) * C + (1/dim) * np.outer(d, d)

                    # Increase sigma slightly
                    sigma *= 1.1

                    # Adapt step size based on success rate
                    if trials >= 10:
                        rate = successes / trials
                        if rate > 0.6:
                            step_size *= 1.05
                        elif rate < 0.4:
                            step_size *= 0.95
                        successes = 0
                        trials = 0
                else:
                    # Failure: decrease sigma
                    sigma *= 0.9
                    # Slightly shrink covariance
                    C = 0.99 * C

                # Ensure sigma stays in reasonable range
                sigma = np.clip(sigma, 0.01, 5.0)

                # Check if we're stuck (no improvement in many iterations)
                if trials % 50 == 0 and trials > 50:
                    # Try a larger jump to escape local minima
                    if time_left() and calls_made < budget - 10:
                        jump = x + rng.normal(0, 2, dim)
                        jump = np.clip(jump, -5, 5)
                        jump, fj = eval_point(jump)
                        if fj < fx:
                            x, fx = jump, fj
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                            sigma = 1.0
                        else:
                            sigma *= 0.7

            # End of this start
            if not time_left():
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
