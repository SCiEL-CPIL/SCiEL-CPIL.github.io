"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Powell's method with multiple restarts and adaptive direction set.
    Uses the conjugate-direction update rule to handle anisotropic curvature.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        calls_made = 0

        def _call(x):
            nonlocal calls_made
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            calls_made += 1
            return val

        # Initial broad sampling to find a good starting point
        n_init = min(50, budget // 10)
        init_points = rng.uniform(-4, 4, size=(n_init, dim))
        for i in range(n_init):
            f_val = _call(init_points[i])
            if f_val < best_f:
                best_f = f_val
                best_x = init_points[i].copy()

        # Powell's method implementation
        def powell_minimize(x0, f0, max_calls, rng_local):
            """
            Powell's method: maintains a set of directions and finds minima along each.
            Returns (best_x, best_f, calls_used)
            """
            dim = self.dim
            x = x0.copy()
            fx = f0
            calls_used = 0

            # Initial directions: standard basis scaled
            directions = np.eye(dim)
            # We'll store directions as a list of vectors
            dirs = [np.eye(dim)[i] for i in range(dim)]

            # Current best along each direction
            x_prev = x.copy()

            while calls_used < max_calls:
                x_old = x.copy()
                f_old = fx
                f_min_dir = np.inf
                i_min = 0

                for i in range(dim):
                    # Minimize along direction i from current x
                    d = dirs[i]
                    # Find optimal step via golden section or parabolic fit
                    # First, do a few evaluations to bracket
                    # Simple approach: try a few step sizes
                    best_step = 0.0
                    best_f_dir = fx
                    # Initial step size estimate
                    step = 0.5
                    # Evaluate at x + step * d
                    x_trial = np.clip(x + step * d, -5, 5)
                    f_trial = func(x_trial)
                    calls_used += 1

                    if f_trial < best_f_dir:
                        best_f_dir = f_trial
                        best_step = step
                        # Try to extend
                        step2 = step * 2
                        x_trial2 = np.clip(x + step2 * d, -5, 5)
                        f_trial2 = func(x_trial2)
                        calls_used += 1
                        if f_trial2 < best_f_dir:
                            best_f_dir = f_trial2
                            best_step = step2
                            # Try more
                            step3 = step2 * 2
                            if calls_used < max_calls:
                                x_trial3 = np.clip(x + step3 * d, -5, 5)
                                f_trial3 = func(x_trial3)
                                calls_used += 1
                                if f_trial3 < best_f_dir:
                                    best_f_dir = f_trial3
                                    best_step = step3
                    else:
                        # Try negative direction
                        step = -0.5
                        x_trial = np.clip(x + step * d, -5, 5)
                        f_trial = func(x_trial)
                        calls_used += 1
                        if f_trial < best_f_dir:
                            best_f_dir = f_trial
                            best_step = step
                            step2 = step * 2
                            if calls_used < max_calls:
                                x_trial2 = np.clip(x + step2 * d, -5, 5)
                                f_trial2 = func(x_trial2)
                                calls_used += 1
                                if f_trial2 < best_f_dir:
                                    best_f_dir = f_trial2
                                    best_step = step2

                    # Parabolic interpolation for better step
                    if best_step != 0.0 and calls_used < max_calls:
                        # Use three points: x, x+best_step*d, and one more
                        # For simplicity, refine with golden section around best_step
                        # This is getting complex; use a simpler refinement
                        # Try a few nearby steps
                        for frac in [0.5, 1.5, 0.25, 2.0]:
                            if calls_used >= max_calls:
                                break
                            trial_step = best_step * frac
                            x_t = np.clip(x + trial_step * d, -5, 5)
                            f_t = func(x_t)
                            calls_used += 1
                            if f_t < best_f_dir:
                                best_f_dir = f_t
                                best_step = trial_step

                    if best_f_dir < f_min_dir:
                        f_min_dir = best_f_dir
                        i_min = i

                    # Update x to the best along this direction
                    x = np.clip(x + best_step * d, -5, 5)
                    fx = best_f_dir

                # Check if we made progress
                if fx >= f_old - 1e-12:
                    break

                # Powell's conjugate direction update
                x_new = x
                f_new = fx

                # Compute the reflected point
                d_new = x_new - x_old
                x_reflected = 2 * x_new - x_old

                if calls_used < max_calls:
                    f_reflected = func(x_reflected)
                    calls_used += 1
                else:
                    break

                if f_reflected < f_old:
                    # Replace the worst direction with the new conjugate direction
                    dirs[i_min] = d_new
                    # Swap to keep the best direction last
                    dirs[i_min], dirs[-1] = dirs[-1], dirs[i_min]
                    x = x_reflected
                    fx = f_reflected
                else:
                    # No improvement from reflection, but we still replaced
                    dirs[i_min] = d_new
                    dirs[i_min], dirs[-1] = dirs[-1], dirs[i_min]

                if time.time() > deadline:
                    break

            return x, fx, calls_used

        # Multiple restarts with Powell's method
        n_restarts = 5
        calls_per_restart = (budget - calls_made) // n_restarts
        if calls_per_restart < 20:
            n_restarts = 1
            calls_per_restart = budget - calls_made

        for r in range(n_restarts):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break

            # For first restart, use the best found point
            if r == 0:
                x_start = best_x.copy()
                f_start = best_f
            else:
                # New random start, biased toward previous best
                if rng.uniform() < 0.5:
                    x_start = best_x + rng.normal(0, 1.0, dim)
                else:
                    x_start = rng.uniform(-4, 4, dim)
                x_start = np.clip(x_start, -5, 5)
                f_start = _call(x_start)
                if f_start < best_f:
                    best_f = f_start
                    best_x = x_start.copy()

            # Run Powell's method
            x_pow, f_pow, calls_used = powell_minimize(
                x_start, f_start, calls_per_restart, rng
            )
            calls_made += calls_used

            if f_pow < best_f:
                best_f = f_pow
                best_x = x_pow.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
