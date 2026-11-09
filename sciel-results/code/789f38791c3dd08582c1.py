"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with diverse initialization and adaptive sigma."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # 10s safety margin before 330s kill
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Track global best
        best_x = None
        best_f = float('inf')

        # Define initial points: origin, random points, and points spread across the box
        # For Rastrigin, the global minimum is at the origin (after transformation),
        # so starting at origin is crucial.
        init_points = [np.zeros(dim)]
        # Add some random points
        for _ in range(7):
            init_points.append(rng.uniform(-4, 4, dim))
        # Add points at specific locations
        for i in range(dim):
            p = np.zeros(dim)
            p[i] = 2.0
            init_points.append(p)
            p[i] = -2.0
            init_points.append(p)

        # Shuffle initial points but keep origin first (or near first) for early progress
        # Actually, let's interleave: origin first, then others
        rng.shuffle(init_points[1:])
        
        calls_made = 0
        max_calls = budget

        for init_idx, x0 in enumerate(init_points):
            if calls_made >= max_calls:
                break
            if time.time() > deadline:
                break

            x = np.clip(x0.copy(), -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # Adaptive sigma for this restart
            sigma = 1.5 if init_idx == 0 else 1.0
            no_improve = 0
            max_no_improve = 50

            while calls_made < max_calls:
                if time.time() > deadline:
                    break

                # Generate candidate
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < fx:
                    x = y
                    fx = fy
                    no_improve = 0
                    # Increase sigma on success
                    sigma = min(sigma * 1.2, 5.0)
                else:
                    no_improve += 1
                    # Decrease sigma on failure
                    sigma = max(sigma * 0.85, 0.01)

                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

                # If stuck for too long, break to next restart
                if no_improve >= max_no_improve:
                    break

                # If sigma is too small, break
                if sigma < 0.01:
                    break

        return best_x, best_f
# EVOLVE-BLOCK-END
