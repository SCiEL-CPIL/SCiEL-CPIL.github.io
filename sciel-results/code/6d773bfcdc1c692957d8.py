"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Derivative-free optimizer using Powell's method with coordinate-wise line search."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 280s out of 300s limit, 20s safety margin
        calls_used = 0

        def safe_call(x):
            nonlocal calls_used
            if calls_used >= self.budget:
                return None
            calls_used += 1
            return float(func(np.asarray(x, dtype=float)))

        # Initialize at origin (minimizer is in [-4,4]^5, origin is a good starting point)
        x = np.zeros(self.dim)
        fx = safe_call(x)
        if fx is None:
            return x, fx

        # Initial directions: standard basis vectors
        directions = [np.eye(self.dim, self.dim)[i] for i in range(self.dim)]

        # Powell's method with coordinate-wise line search
        # For each direction, we do a 1D line search along that direction from current point
        # We use a simple ternary search or golden section for the 1D minimization

        def line_search_1d(base, direction, fval_base):
            """
            1D line search along direction from base point.
            Uses golden section search to find minimum along the line.
            Returns (best_point, best_fval, evals_used)
            """
            # First, find a bracket [lo, hi] where the minimum lies
            # Start with step size 1.0 and expand if needed
            step = 1.0
            lo = 0.0
            hi = 0.0
            f_lo = fval_base

            # Try positive direction first
            x_plus = base + step * direction
            f_plus = safe_call(x_plus)
            if f_plus is None:
                return base, fval_base, 1

            if f_plus < f_lo:
                # Minimum is in positive direction, expand bracket
                lo = 0.0
                hi = step
                f_hi = f_plus
                # Try to expand further
                while True:
                    x_new = base + 2 * step * direction
                    f_new = safe_call(x_new)
                    if f_new is None or f_new >= f_hi:
                        break
                    lo = hi
                    hi = 2 * step
                    f_hi = f_new
                    step *= 2
                    # Check bounds
                    if np.any(x_new < -5.5) or np.any(x_new > 5.5):
                        break
            else:
                # Try negative direction
                x_minus = base - step * direction
                f_minus = safe_call(x_minus)
                if f_minus is None:
                    return base, fval_base, 2

                if f_minus < f_lo:
                    # Minimum is in negative direction
                    hi = 0.0
                    lo = -step
                    f_lo_new = f_minus
                    # Expand in negative direction
                    while True:
                        x_new = base - 2 * step * direction
                        f_new = safe_call(x_new)
                        if f_new is None or f_new >= f_lo_new:
                            break
                        hi = lo
                        lo = -2 * step
                        f_lo_new = f_new
                        step *= 2
                        if np.any(x_new < -5.5) or np.any(x_new > 5.5):
                            break
                    # Now bracket is [lo, 0] with f(lo) < f(0)
                    # Use golden section on [lo, 0]
                    a, b = lo, 0.0
                    fa, fb = f_lo_new, fval_base
                else:
                    # No improvement, return base
                    return base, fval_base, 3

            # If we have a bracket [lo, hi], do golden section search
            # But we need to be careful: the bracket might be [lo, 0] or [0, hi]
            # Let's just do a simple approach: try several points and keep the best

            # Actually, let's do a more robust approach:
            # Sample points along the line and keep the best, then refine
            best_x = base
            best_f = fval_base

            # Check positive side
            for s in [0.5, 1.0, 2.0, 4.0]:
                x_try = base + s * direction
                f_try = safe_call(x_try)
                if f_try is None:
                    break
                if f_try < best_f:
                    best_x = x_try.copy()
                    best_f = f_try

            # Check negative side
            for s in [0.5, 1.0, 2.0, 4.0]:
                x_try = base - s * direction
                f_try = safe_call(x_try)
                if f_try is None:
                    break
                if f_try < best_f:
                    best_x = x_try.copy()
                    best_f = f_try

            # Refine around best point with smaller steps
            if best_x is not base:
                # Determine direction of best improvement
                delta_vec = best_x - base
                norm_delta = np.linalg.norm(delta_vec)
                if norm_delta > 1e-10:
                    dir_vec = delta_vec / norm_delta
                    # Try steps around the best found
                    for s in [0.25, 0.75, 1.25, 1.5]:
                        x_try = base + s * norm_delta * dir_vec
                        f_try = safe_call(x_try)
                        if f_try is not None and f_try < best_f:
                            best_x = x_try.copy()
                            best_f = f_try

            return best_x, best_f, calls_used

        # Main Powell loop
        tol = 1e-10
        max_powell_iters = 100

        for powell_iter in range(max_powell_iters):
            if time.time() > deadline - 5.0:
                break
            if calls_used >= self.budget:
                break

            fx_old = fx
            improved = False

            for d in range(self.dim):
                if time.time() > deadline - 5.0:
                    break
                if calls_used >= self.budget:
                    break

                # Line search along directions[d]
                x_new, f_new, _ = line_search_1d(x, directions[d], fx)

                if f_new < fx:
                    x = x_new
                    fx = f_new
                    improved = True

            # Check convergence
            if not improved or abs(fx_old - fx) < tol:
                break

            # Update directions: replace the last direction with the vector from start to end
            # We need to track the start of this iteration
            # Actually, let's do a simpler version: just keep doing coordinate searches
            # and periodically add the overall displacement as a new direction

            # Replace the direction that gave the largest improvement with the displacement
            # For simplicity, let's just cycle through and occasionally update

            if powell_iter % 5 == 4 and powell_iter > 0:
                # Add a new direction based on recent displacement
                # We don't track displacement easily, so let's just rotate directions
                directions = directions[1:] + [directions[0]]

        return x, fx
# EVOLVE-BLOCK-END
