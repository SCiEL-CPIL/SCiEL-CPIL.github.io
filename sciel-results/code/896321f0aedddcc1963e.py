"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Pure Random Coordinate Descent (RCD) optimizer for BBOB f9."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 330s kill
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Initialize at origin
        best_x = np.zeros(n)
        best_f = func(best_x)
        calls_used = 1

        def check_deadline():
            return time.time() >= deadline

        def do_call(x):
            nonlocal calls_used
            fx = func(x)
            calls_used += 1
            return fx

        # RCD strategy:
        # - Maintain a current point x and its function value fx
        # - At each step, pick a random coordinate, perturb it by +/- step_size
        # - If improvement, accept and shrink step size slightly
        # - If no improvement, reject and shrink step size more
        # - Periodically restart from best_x with larger step to escape local minima

        x = best_x.copy()
        fx = best_f
        step_size = 2.0
        min_step = 1e-10
        max_step = 8.0
        shrink_success = 0.95   # shrink factor on success
        shrink_fail = 0.85      # shrink factor on failure
        restart_every = 50      # restart frequency in steps
        steps_since_restart = 0

        total_steps = 0
        while calls_used < budget and not check_deadline():
            # Pick a random coordinate
            coord = rng.integers(0, n)
            
            # Generate two candidates: +step and -step along that coordinate
            y_plus = x.copy()
            y_plus[coord] = np.clip(y_plus[coord] + step_size, -5, 5)
            y_minus = x.copy()
            y_minus[coord] = np.clip(y_minus[coord] - step_size, -5, 5)

            # Evaluate both
            f_plus = do_call(y_plus)
            if calls_used >= budget or check_deadline():
                break
            f_minus = do_call(y_minus)

            # Pick the better candidate
            if f_plus <= f_minus:
                y_new = y_plus
                f_new = f_plus
            else:
                y_new = y_minus
                f_new = f_minus

            if f_new < fx:
                # Accept
                x = y_new
                fx = f_new
                step_size = max(step_size * shrink_success, min_step)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                steps_since_restart = 0
            else:
                # Reject
                step_size = max(step_size * shrink_fail, min_step)
                steps_since_restart += 1

            total_steps += 2  # two function calls per iteration

            # Periodic restart from best with larger step to escape narrow valleys
            if steps_since_restart >= restart_every:
                x = best_x.copy() + 1.0 * rng.standard_normal(n)
                x = np.clip(x, -5, 5)
                fx = do_call(x)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                step_size = min(max(step_size * 1.5, 0.5), max_step)
                steps_since_restart = 0
                total_steps += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
