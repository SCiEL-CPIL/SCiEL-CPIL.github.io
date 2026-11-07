"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive Derandomized Coordinate Descent with per-axis step scaling."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        # Per-coordinate step sizes
        steps = np.ones(dim) * 1.0
        # Success counters for adaptive step scaling
        success_count = np.zeros(dim, dtype=int)
        window_size = 10

        calls_made = 1

        # Phase 1: Coordinate descent with adaptive steps
        # We cycle through coordinates, trying +/- moves
        i = 0
        while calls_made < budget:
            if time.time() > deadline:
                break

            # Check for time pressure
            remaining_time = deadline - time.time()
            if remaining_time < 5.0:
                break

            d = i % dim
            i += 1

            # Try positive and negative moves along axis d
            for sign in [1.0, -1.0]:
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break

                x_trial = x.copy()
                x_trial[d] = x[d] + sign * steps[d]
                x_trial = np.clip(x_trial, -5, 5)

                # Skip if no change
                if np.array_equal(x_trial, x):
                    continue

                fy = func(x_trial)
                calls_made += 1

                if fy < fx:
                    # Success: move and increase step
                    x = x_trial
                    fx = fy
                    success_count[d] += 1
                    steps[d] *= 1.3

                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    # Failure: shrink step
                    steps[d] *= 0.7

            # Reset success counter periodically
            if i % (dim * window_size) == 0:
                success_count *= 0.5

            # Adaptive step size based on success rate
            for d2 in range(dim):
                rate = success_count[d2] / (i // dim + 1e-10)
                if rate > 0.8:
                    steps[d2] = min(steps[d2] * 1.1, 10.0)
                elif rate < 0.2:
                    steps[d2] = max(steps[d2] * 0.9, 0.01)

        # Phase 2: Fine-tuning with smaller steps if budget remains
        if calls_made < budget:
            fine_steps = steps * 0.1
            for _ in range(3):
                for d in range(dim):
                    if calls_made >= budget:
                        break
                    if time.time() > deadline:
                        break
                    for sign in [1.0, -1.0]:
                        if calls_made >= budget:
                            break
                        if time.time() > deadline:
                            break
                        x_trial = x.copy()
                        x_trial[d] = x[d] + sign * fine_steps[d]
                        x_trial = np.clip(x_trial, -5, 5)
                        if np.array_equal(x_trial, x):
                            continue
                        fy = func(x_trial)
                        calls_made += 1
                        if fy < fx:
                            x = x_trial
                            fx = fy
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
