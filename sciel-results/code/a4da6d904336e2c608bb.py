"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Deterministic Hooke-Jeeves pattern search with deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 280.0  # safety margin before 330s kill
        calls = 0
        budget = self.budget
        dim = self.dim

        # Initial point
        x = np.zeros(dim)
        fx = func(x)
        calls += 1
        best_x, best_f = x.copy(), fx

        step = 1.0
        direction = None
        consecutive_failures = 0
        max_consecutive_failures = 8

        while calls < budget:
            if time.time() > deadline:
                break

            # Exploratory phase: try each coordinate
            improved = False
            for i in range(dim):
                if calls >= budget or time.time() > deadline:
                    break

                # Try positive step
                x_try = x.copy()
                x_try[i] += step
                x_try = np.clip(x_try, -5, 5)
                f_try = func(x_try)
                calls += 1

                if f_try < fx:
                    x = x_try
                    fx = f_try
                    improved = True
                    if fx < best_f:
                        best_x, best_f = x.copy(), fx
                    direction = np.zeros(dim)
                    direction[i] = 1.0
                    break  # move to next coordinate

                # Try negative step
                x_try = x.copy()
                x_try[i] -= step
                x_try = np.clip(x_try, -5, 5)
                f_try = func(x_try)
                calls += 1

                if f_try < fx:
                    x = x_try
                    fx = f_try
                    improved = True
                    if fx < best_f:
                        best_x, best_f = x.copy(), fx
                    direction = np.zeros(dim)
                    direction[i] = -1.0
                    break  # move to next coordinate

            if not improved:
                consecutive_failures += 1
                if consecutive_failures >= max_consecutive_failures:
                    # Halve the step and try again
                    step *= 0.5
                    consecutive_failures = 0
                    if step < 1e-10:
                        break
                continue
            else:
                consecutive_failures = 0

            # Pattern step: try to extend along the last successful direction
            if direction is not None and calls < budget:
                x_pattern = x + direction * step
                x_pattern = np.clip(x_pattern, -5, 5)
                f_pattern = func(x_pattern)
                calls += 1

                if f_pattern < fx:
                    # Keep stepping along this direction
                    while calls < budget:
                        x_next = x_pattern + direction * step
                        x_next = np.clip(x_next, -5, 5)
                        f_next = func(x_next)
                        calls += 1
                        if f_next < f_pattern:
                            x_pattern = x_next
                            f_pattern = f_next
                        else:
                            break
                    x, fx = x_pattern, f_pattern
                    if fx < best_f:
                        best_x, best_f = x.copy(), fx
                    if time.time() > deadline:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
