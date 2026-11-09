"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Multi-start L-BFGS-B local search for rotated Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        # Track best solution found
        best_x = np.clip(rng.uniform(-5, 5, self.dim), -5, 5)
        best_f = func(best_x)
        calls_made = 1

        # Define a wrapper that tracks calls and enforces the budget
        call_count = [0]
        def tracked_func(x):
            if call_count[0] >= self.budget:
                return best_f  # Return current best to stop optimization
            call_count[0] += 1
            calls_made[0] = call_count[0]
            return func(np.clip(x, -5, 5))

        # Generate diverse starting points
        # Mix of random points and points near the origin
        n_starts = 15
        starts = []
        # A few points near the center
        for i in range(5):
            starts.append(rng.uniform(-2, 2, self.dim))
        # Random points spread across the domain
        for i in range(10):
            starts.append(rng.uniform(-5, 5, self.dim))
        # A few points at the corners
        for i in range(3):
            corner = np.array([rng.choice([-5, 5]) for _ in range(self.dim)])
            starts.append(corner)

        # Run L-BFGS-B from each start point
        for i, x0 in enumerate(starts):
            if call_count[0] >= self.budget:
                break
            if time.time() > deadline:
                break

            try:
                result = minimize(
                    tracked_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * self.dim,
                    options={
                        'maxiter': 200,
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                        'maxls': 20
                    }
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x
            except Exception:
                pass

            # If we've used more than half the budget, reduce the number of remaining starts
            if call_count[0] > self.budget * 0.5 and i < n_starts - 1:
                # For remaining starts, use a smaller maxiter
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
