"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Cyclic Adaptive Coordinate Descent for BBOB-style f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        calls_used = 0
        n = self.dim

        def safe_call(x):
            nonlocal calls_used
            if calls_used >= self.budget:
                return None
            calls_used += 1
            return float(func(np.asarray(x, dtype=float)))

        # Cache of evaluated points to avoid re-evaluation
        eval_cache = {}

        def get_f(x):
            key = tuple(np.round(x, 10))
            if key in eval_cache:
                return eval_cache[key]
            f = safe_call(x)
            if f is not None:
                eval_cache[key] = f
            return f

        # Initialize at origin
        x = np.zeros(n)
        fx = safe_call(x)
        if fx is None:
            return x, fx

        best_x = x.copy()
        best_f = fx

        # Adaptive step sizes per dimension
        step_sizes = np.ones(n) * 1.0
        min_step = 1e-6
        max_step = 10.0

        # Cyclic coordinate descent with adaptive step size
        while calls_used < self.budget:
            if time.time() > deadline - 5.0:
                break

            for dim_i in range(n):
                if calls_used >= self.budget or time.time() > deadline - 5.0:
                    break

                # Try positive and negative steps
                best_local_f = fx
                best_local_offset = 0.0
                current_step = step_sizes[dim_i]

                # Adaptive line search: try several step sizes
                for factor in [1.0, 0.5, 0.25, 2.0, 4.0]:
                    trial_step = current_step * factor
                    if trial_step < min_step or trial_step > max_step:
                        continue

                    # Positive direction
                    x_trial = x.copy()
                    x_trial[dim_i] += trial_step
                    f_trial = get_f(x_trial)
                    if f_trial is not None and f_trial < best_local_f:
                        best_local_f = f_trial
                        best_local_offset = trial_step

                    # Negative direction
                    x_trial = x.copy()
                    x_trial[dim_i] -= trial_step
                    f_trial = get_f(x_trial)
                    if f_trial is not None and f_trial < best_local_f:
                        best_local_f = f_trial
                        best_local_offset = -trial_step

                # Update position and step size
                if best_local_offset != 0.0:
                    x[dim_i] += best_local_offset
                    fx = best_local_f
                    # Increase step size if we made progress
                    step_sizes[dim_i] = min(max_step, step_sizes[dim_i] * 1.5)
                else:
                    # Decrease step size if no progress
                    step_sizes[dim_i] = max(min_step, step_sizes[dim_i] * 0.5)

                # Update global best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
