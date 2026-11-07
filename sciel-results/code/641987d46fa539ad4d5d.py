"""Derivative-Free Coordinate Descent with multi-resolution grid for Lunacek bi-Rastrigin."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-resolution coordinate descent optimizer for bi-modal functions."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls = 0

        best_x = np.zeros(dim)
        best_fx = func(best_x)
        calls += 1

        def try_point(x):
            nonlocal best_x, best_fx, calls
            if calls >= budget:
                return
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_x = x.copy()
                best_fx = fx

        def coordinate_descent(x, step_sizes, max_rounds=None):
            """Perform coordinate descent with given step sizes."""
            nonlocal calls
            x = x.copy()
            fx = func(x)
            calls += 1
            if fx < best_fx:
                best_x = x.copy()
                best_fx = fx

            for step in step_sizes:
                if calls >= budget or time.time() > deadline:
                    break
                for _ in range(10):  # 10 passes per step size
                    if calls >= budget or time.time() > deadline:
                        break
                    improved = False
                    for d in range(dim):
                        if calls >= budget or time.time() > deadline:
                            break
                        # Try +step
                        x_try = x.copy()
                        x_try[d] += step
                        x_try = np.clip(x_try, -5, 5)
                        fy = func(x_try)
                        calls += 1
                        if fy < fx:
                            x = x_try
                            fx = fy
                            improved = True
                            if fx < best_fx:
                                best_x = x.copy()
                                best_fx = fx
                        else:
                            # Try -step
                            x_try = x.copy()
                            x_try[d] -= step
                            x_try = np.clip(x_try, -5, 5)
                            fy = func(x_try)
                            calls += 1
                            if fy < fx:
                                x = x_try
                                fx = fy
                                improved = True
                                if fx < best_fx:
                                    best_x = x.copy()
                                    best_fx = fx
                    if not improved:
                        break

        # Phase 1: Coarse grid scan to identify promising regions
        # Lunacek bi-Rastrigin has a global minimum near center and a local minimum offset
        # Try several starting points: center, and offset points

        # Starting points for multi-start
        start_points = [
            np.zeros(dim),
            np.array([1.0] * dim),
            np.array([-1.0] * dim),
            np.array([2.0] * dim),
            np.array([-2.0] * dim),
            np.array([3.0] * dim),
            np.array([-3.0] * dim),
        ]

        # Coarse step sizes for initial exploration
        coarse_steps = [2.0, 1.0, 0.5, 0.25, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

        # Do coordinate descent from each starting point with a portion of the budget
        n_starts = min(len(start_points), 5)
        budget_per_start = max(50, budget // n_starts)

        for i in range(n_starts):
            if calls >= budget or time.time() > deadline:
                break
            x0 = start_points[i].copy()
            # Use a subset of step sizes for each start
            steps_for_this = coarse_steps[:max(3, len(coarse_steps) - i)]
            coordinate_descent(x0, steps_for_this)

        # Phase 2: Fine-tune from the best point found so far
        if calls < budget and time.time() < deadline and best_x is not None:
            fine_steps = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]
            coordinate_descent(best_x, fine_steps)

        # Phase 3: If budget remains, do another round from best with even finer steps
        if calls < budget and time.time() < deadline and best_x is not None:
            ultra_fine_steps = [0.0005, 0.0002, 0.0001, 0.00005, 0.00002, 0.00001]
            coordinate_descent(best_x, ultra_fine_steps)

        return best_x, best_fx
# EVOLVE-BLOCK-END
