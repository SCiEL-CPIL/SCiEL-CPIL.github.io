"""Deterministic Hooke-Jeeves pattern search with deadline guard."""
import numpy as np
import time


class Optimizer:
    """Hooke-Jeeves pattern search for box-constrained minimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls = 0

        # Initial point: center of the box
        x = np.zeros(dim)
        fx = func(x)
        calls += 1

        best_x = x.copy()
        best_fx = fx

        # Initial step size
        delta = 2.0

        # Track last successful direction for pattern step
        last_direction = None
        pattern_successes = 0

        while calls < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Phase 1: Exploratory search
            improved = False
            x_new = x.copy()
            for i in range(dim):
                if time.time() > deadline:
                    break
                if calls >= budget:
                    break

                # Try positive step
                trial = x_new.copy()
                trial[i] += delta
                trial = np.clip(trial, -5, 5)
                ft = func(trial)
                calls += 1

                if ft < fx:
                    x_new = trial
                    fx = ft
                    improved = True

                # Try negative step
                trial = x_new.copy()
                trial[i] -= delta
                trial = np.clip(trial, -5, 5)
                ft = func(trial)
                calls += 1

                if ft < fx:
                    x_new = trial
                    fx = ft
                    improved = True

            if improved:
                # Update best
                if fx < best_fx:
                    best_fx = fx
                    best_x = x_new.copy()

                # Record direction for pattern step
                direction = x_new - x
                x = x_new

                # Phase 2: Pattern step
                # Try to accelerate along the last direction
                x_pattern = x + direction
                x_pattern = np.clip(x_pattern, -5, 5)
                fp = func(x_pattern)
                calls += 1

                if fp < fx:
                    # Pattern step successful
                    x = x_pattern
                    fx = fp
                    if fx < best_fx:
                        best_fx = fx
                        best_x = x.copy()
                    pattern_successes += 1

                    # Try to extend the pattern step further
                    while calls < budget and time.time() < deadline:
                        x_ext = x + direction
                        x_ext = np.clip(x_ext, -5, 5)
                        # Check if we're making progress (not just clipping)
                        if np.allclose(x_ext, x, atol=1e-10):
                            break
                        fe = func(x_ext)
                        calls += 1
                        if fe < fx:
                            x = x_ext
                            fx = fe
                            if fx < best_fx:
                                best_fx = fx
                                best_x = x.copy()
                            pattern_successes += 1
                        else:
                            break
                else:
                    # Pattern step failed, reset
                    pattern_successes = 0
                    direction = None
            else:
                # No improvement in exploratory phase
                if pattern_successes > 0:
                    # We had pattern successes before, try shrinking
                    delta *= 0.5
                    pattern_successes = 0
                else:
                    delta *= 0.5

                # If delta is too small, we're essentially done
                if delta < 1e-6:
                    break

        return best_x, best_fx
