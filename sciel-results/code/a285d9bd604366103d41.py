import numpy as np
import time

class Optimizer:
    """
    Optimizer using Adaptive Coordinate Descent for BBOB f13 (Sharp Ridge).
    
    The Sharp Ridge function has anisotropic geometry: narrow in one direction,
    wide in another, with a sharp quadratic wall. Coordinate descent with
    adaptive per-axis step sizes exploits this structure by:
    - Optimizing each dimension independently via 1D line search
    - Dynamically scaling steps based on recent progress along each axis
    - Growing steps when making progress, shrinking when stuck
    - Periodic restarts to escape local minima
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # Safety margin for 330s limit

        calls = 0
        best_x = None
        best_f = float('inf')
        rng = np.random.default_rng(12345)

        def safe_call(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return None
            if time.time() > deadline:
                return None
            calls += 1
            try:
                x_clipped = np.clip(x, -5.0, 5.0)
                f_val = func(x_clipped)
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_clipped.copy()
                return f_val
            except Exception:
                return None

        # Phase 1: Global Exploration - sample to find the basin
        n_random = 15
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, size=dim)
            f_val = safe_call(x)
            if f_val is None:
                break

        if best_x is None:
            x0 = np.zeros(dim)
            f0 = safe_call(x0)
            if f0 is None:
                return best_x, best_f
            best_x = x0.copy()
            best_f = f0

        # Phase 2: Adaptive Coordinate Descent
        # Per-axis step sizes, initialized to cover the domain
        step_sizes = np.full(dim, 1.0)
        # Track recent improvement per axis to adapt steps
        recent_improvement = np.zeros(dim)
        # Golden ratio for 1D search
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        inv_phi = 1.0 / phi

        def golden_section_1d(x, axis, step, f_start):
            """
            1D golden-section search along a single axis.
            Returns best_x_1d, best_f_1d, best_offset.
            """
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return x.copy(), f_start, 0.0

            # Initial bracket: try +/- step
            a = -step
            b = step
            x_a = x.copy()
            x_a[axis] += a
            f_a = safe_call(x_a)
            if f_a is None:
                return x.copy(), f_start, 0.0

            x_b = x.copy()
            x_b[axis] += b
            f_b = safe_call(x_b)
            if f_b is None:
                return x.copy(), f_start, 0.0

            # If neither endpoint is better, try smaller steps
            if f_a >= f_start and f_b >= f_start:
                # Try half-step
                a = -step * 0.5
                b = step * 0.5
                x_a = x.copy()
                x_a[axis] += a
                f_a = safe_call(x_a)
                if f_a is None:
                    return x.copy(), f_start, 0.0
                x_b = x.copy()
                x_b[axis] += b
                f_b = safe_call(x_b)
                if f_b is None:
                    return x.copy(), f_start, 0.0
                if f_a >= f_start and f_b >= f_start:
                    # Try quarter-step
                    a = -step * 0.25
                    b = step * 0.25
                    x_a = x.copy()
                    x_a[axis] += a
                    f_a = safe_call(x_a)
                    if f_a is None:
                        return x.copy(), f_start, 0.0
                    x_b = x.copy()
                    x_b[axis] += b
                    f_b = safe_call(x_b)
                    if f_b is None:
                        return x.copy(), f_start, 0.0
                    if f_a >= f_start and f_b >= f_start:
                        return x.copy(), f_start, 0.0

            # Golden section search within [a, b]
            # Ensure a < b
            if a > b:
                a, b = b, a
                f_a, f_b = f_b, f_a

            # We need two interior points
            # d = b - a
            d = b - a
            if d < 1e-12:
                return x.copy(), f_start, 0.0

            # Interior points
            c = b - inv_phi * d
            w = a + inv_phi * d

            x_c = x.copy()
            x_c[axis] += c
            f_c = safe_call(x_c)
            if f_c is None:
                return x.copy(), f_start, 0.0

            x_w = x.copy()
            x_w[axis] += w
            f_w = safe_call(x_w)
            if f_w is None:
                return x.copy(), f_start, 0.0

            # Track best so far
            best_offset = 0.0
            best_f_1d = f_start
            best_x_1d = x.copy()

            # Check endpoints and interior
            if f_a < best_f_1d:
                best_f_1d = f_a
                best_offset = a
                best_x_1d = x.copy()
                best_x_1d[axis] += a
            if f_b < best_f_1d:
                best_f_1d = f_b
                best_offset = b
                best_x_1d = x.copy()
                best_x_1d[axis] += b
            if f_c < best_f_1d:
                best_f_1d = f_c
                best_offset = c
                best_x_1d = x.copy()
                best_x_1d[axis] += c
            if f_w < best_f_1d:
                best_f_1d = f_w
                best_offset = w
                best_x_1d = x.copy()
                best_x_1d[axis] += w

            # Golden section iterations
            max_gs_iter = 12
            for _ in range(max_gs_iter):
                if calls >= budget or time.time() > deadline:
                    break

                d = b - a
                if d < 1e-10:
                    break

                if f_c < f_w:
                    # Minimum is in [a, w]
                    b = w
                    f_b = f_w
                    w = c
                    f_w = f_c
                    c = b - inv_phi * (b - a)
                    x_c = x.copy()
                    x_c[axis] += c
                    f_c = safe_call(x_c)
                    if f_c is None:
                        break
                    if f_c < best_f_1d:
                        best_f_1d = f_c
                        best_offset = c
                        best_x_1d = x.copy()
                        best_x_1d[axis] += c
                else:
                    # Minimum is in [c, b]
                    a = c
                    f_a = f_c
                    c = w
                    f_c = f_w
                    w = a + inv_phi * (b - a)
                    x_w = x.copy()
                    x_w[axis] += w
                    f_w = safe_call(x_w)
                    if f_w is None:
                        break
                    if f_w < best_f_1d:
                        best_f_1d = f_w
                        best_offset = w
                        best_x_1d = x.copy()
                        best_x_1d[axis] += w

            # Update global best
            if best_f_1d < best_f:
                best_f = best_f_1d
                best_x = best_x_1d.copy()

            return best_x_1d, best_f_1d, best_offset

        # Main coordinate descent loop
        stagnation_counter = 0
        prev_best = best_f
        max_stagnation = 8  # number of full axis sweeps without improvement

        while calls < budget and time.time() < deadline:
            # One full sweep over all axes
            sweep_start_f = best_f
            for axis in range(dim):
                if calls >= budget or time.time() > deadline:
                    break

                # Adaptive step size: grow if recent improvement, shrink if not
                if recent_improvement[axis] > 1e-6:
                    step_sizes[axis] = min(step_sizes[axis] * 1.5, 5.0)
                else:
                    step_sizes[axis] = max(step_sizes[axis] * 0.7, 0.01)

                # Perform 1D search along this axis from current best
                x_new, f_new, offset = golden_section_1d(
                    best_x, axis, step_sizes[axis], best_f
                )

                if x_new is not None and f_new < best_f:
                    best_x = x_new
                    best_f = f_new
                    recent_improvement[axis] = sweep_start_f - f_new
                else:
                    recent_improvement[axis] = 0.0

            # Check for stagnation
            if best_f < prev_best - 1e-8:
                prev_best = best_f
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # If stagnant, try a perturbation/restart
            if stagnation_counter >= max_stagnation:
                if calls < budget - 20:
                    # Perturb from current best
                    x_pert = best_x + rng.normal(0, 0.5, size=dim)
                    x_pert = np.clip(x_pert, -5.0, 5.0)
                    f_pert = safe_call(x_pert)
                    if f_pert is not None and f_pert < best_f:
                        best_x = x_pert
                        best_f = f_pert
                        stagnation_counter = 0
                    else:
                        # Try a more aggressive random restart
                        x_rand = rng.uniform(-5.0, 5.0, size=dim)
                        f_rand = safe_call(x_rand)
                        if f_rand is not None and f_rand < best_f:
                            best_x = x_rand
                            best_f = f_rand
                            stagnation_counter = 0
                        else:
                            # Reset step sizes to encourage exploration
                            step_sizes = np.full(dim, 1.0)
                            recent_improvement = np.zeros(dim)
                            stagnation_counter = 0

        return best_x, best_f
