"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coordinate Descent with adaptive step sizes and perturbations."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill

        rng = np.random.default_rng(42)

        # Initialize
        x = rng.uniform(-5, 5, self.dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Per-dimension step sizes
        h = np.ones(self.dim) * 2.0
        min_h = 1e-4
        max_h = 5.0

        no_improve_count = 0
        max_no_improve = 15

        while calls < self.budget:
            if time.time() > deadline:
                break

            moved = False
            # Cyclic descent over dimensions
            for i in range(self.dim):
                if time.time() > deadline:
                    break

                # Try positive step
                x_temp = x.copy()
                x_temp[i] += h[i]
                x_temp = np.clip(x_temp, -5, 5)
                f_temp = func(x_temp)
                calls += 1

                if f_temp < fx:
                    x = x_temp
                    fx = f_temp
                    moved = True
                    h[i] = min(h[i] * 1.1, max_h)
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                    continue

                # Try negative step
                x_temp = x.copy()
                x_temp[i] -= h[i]
                x_temp = np.clip(x_temp, -5, 5)
                f_temp = func(x_temp)
                calls += 1

                if f_temp < fx:
                    x = x_temp
                    fx = f_temp
                    moved = True
                    h[i] = min(h[i] * 1.1, max_h)
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                        no_improve_count = 0
                    else:
                        no_improve_count += 1
                else:
                    h[i] = max(h[i] * 0.7, min_h)
                    no_improve_count += 1

            if not moved:
                # No movement in entire cycle
                h = np.maximum(h * 0.5, min_h)
                if np.max(h) < min_h * 10:
                    # Perturb to escape local minimum
                    x = np.clip(x + rng.normal(0, 1.5, self.dim), -5, 5)
                    fx = func(x)
                    calls += 1
                    h = np.ones(self.dim) * 1.0
                    no_improve_count = 0
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
            else:
                # If we moved but didn't improve global best, check for stagnation
                if no_improve_count > max_no_improve:
                    # Perturb
                    x = np.clip(x + rng.normal(0, 2.0, self.dim), -5, 5)
                    fx = func(x)
                    calls += 1
                    h = np.ones(self.dim) * 1.5
                    no_improve_count = 0
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
