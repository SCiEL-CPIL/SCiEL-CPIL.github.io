"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coordinate-wise line search with adaptive per-dimension step sizes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        n = self.dim

        # Initialize at a random point
        x = rng.uniform(-5, 5, n)
        fx = func(x)

        # Per-dimension step sizes
        sigma = np.ones(n) * 1.0

        # Track consecutive failures per dimension for step size adaptation
        fails = np.zeros(n, dtype=int)

        calls_used = 1

        def time_ok():
            return (time.time() - start_time) < deadline

        while calls_used < self.budget and time_ok():
            # Try each dimension in a random order
            dims = rng.permutation(n)
            for d in dims:
                if calls_used >= self.budget or not time_ok():
                    break

                # Try positive and negative directions
                best_delta = 0.0
                best_val = fx

                # Line search along dimension d
                step = sigma[d]
                for _ in range(4):
                    # Try positive step
                    y_pos = x.copy()
                    y_pos[d] = np.clip(x[d] + step, -5, 5)
                    fy_pos = func(y_pos)
                    calls_used += 1

                    if fy_pos < best_val:
                        best_val = fy_pos
                        best_delta = step
                        step *= 1.6
                    else:
                        # Try negative step
                        y_neg = x.copy()
                        y_neg[d] = np.clip(x[d] - step, -5, 5)
                        fy_neg = func(y_neg)
                        calls_used += 1

                        if fy_neg < best_val:
                            best_val = fy_neg
                            best_delta = -step
                            step *= 1.6
                        else:
                            # No improvement; shrink step
                            step *= 0.5

                    if calls_used >= self.budget or not time_ok():
                        break

                if best_delta != 0.0:
                    x[d] = np.clip(x[d] + best_delta, -5, 5)
                    fx = best_val
                    sigma[d] *= 1.3  # increase step size on success
                    fails[d] = 0
                else:
                    sigma[d] *= 0.7  # decrease step size on failure
                    fails[d] += 1

                    # If too many consecutive failures, restart this dimension's step
                    if fails[d] > 8:
                        sigma[d] = 1.0
                        fails[d] = 0

        return x, fx
# EVOLVE-BLOCK-END
