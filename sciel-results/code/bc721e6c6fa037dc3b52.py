"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, enhanced with restarts and a deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin

        best_x = None
        best_f = np.inf

        # We'll do multiple restarts if time permits, but primarily rely on a single
        # well-tuned (1+1)-ES run since the budget is only 500 evaluations.
        # The key insight: for a step ellipsoidal function, we need to find the
        # discontinuity. A (1+1)-ES with good sigma adaptation can do this.

        # Use a fixed seed for reproducibility
        rng = np.random.default_rng(42)

        # Initial point: try center first, then random
        x = np.zeros(self.dim)
        fx = func(x)
        best_x, best_f = x.copy(), fx
        calls = 1

        sigma = 1.0  # Start with moderate step size
        no_improve = 0
        max_no_improve = 50

        for i in range(self.budget - 1):
            # Check time deadline
            if time.time() > deadline:
                break

            # Generate candidate
            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx:
                x, fx = y, fy
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                no_improve = 0
                # Increase sigma on success (cautiously)
                sigma *= 1.2
            else:
                no_improve += 1
                # Decrease sigma on failure
                sigma *= 0.8

            # If no improvement for a while, do a restart with a larger perturbation
            if no_improve >= max_no_improve:
                # Try a larger jump to escape local region
                sigma = 2.0
                no_improve = 0
                # Also try a random restart from a new point
                if calls < self.budget - 10:
                    x_new = rng.uniform(-5, 5, self.dim)
                    fx_new = func(x_new)
                    calls += 1
                    if fx_new < fx:
                        x, fx = x_new, fx_new
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    sigma = 1.0

            # Clamp sigma to reasonable bounds
            sigma = np.clip(sigma, 0.01, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
