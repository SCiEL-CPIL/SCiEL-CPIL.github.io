"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools


class Optimizer:
    """Deterministic grid search + fixed-step local hill climb."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s kill
        n = self.dim
        budget = self.budget
        calls = 0

        def safe_call(x):
            nonlocal calls
            if calls >= budget:
                return None
            calls += 1
            return func(x)

        # Phase 1: Deterministic grid search
        # Use a 3-level grid: -5, 0, 5 for each dimension
        # This yields 3^5 = 243 points, which is well within the budget of 500
        levels = [-5.0, 0.0, 5.0]
        grid_points = []
        for combo in itertools.product(levels, repeat=n):
            grid_points.append(np.array(combo))

        # Evaluate all grid points
        best_x = None
        best_f = float('inf')
        for v in grid_points:
            if calls >= budget or time.time() > deadline:
                break
            f = safe_call(v)
            if f is not None and f < best_f:
                best_f = f
                best_x = v.copy()

        if best_x is None:
            # Fallback: evaluate at origin
            best_x = np.zeros(n)
            best_f = safe_call(best_x)

        # Phase 2: Fixed-step local hill climb from best point
        # Use a sequence of decreasing step sizes
        # Added smaller steps for final polish
        step_sizes = [2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
        for step in step_sizes:
            improved = True
            while improved and calls < budget and time.time() < deadline:
                improved = False
                for i in range(n):
                    for direction in [-1, 1]:
                        y = best_x.copy()
                        y[i] += direction * step
                        y = np.clip(y, -5, 5)
                        fy = safe_call(y)
                        if fy is not None and fy < best_f:
                            best_x = y
                            best_f = fy
                            improved = True
                            if calls >= budget or time.time() > deadline:
                                break
                    if calls >= budget or time.time() > deadline:
                        break
                if calls >= budget or time.time() > deadline:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
