"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase derivative-free optimizer for Weierstrass f16.

    Phase 1: Structured lattice exploration (Hammersley points) to find good basins.
    Phase 2: Adaptive (1+1)-ES with coordinate-descent polish for local refinement.
    Phase 3: Final high-precision coordinate polish.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285  # 15s safety margin under 300s limit
        rng = np.random.default_rng(42)

        def elapsed_ok():
            return (time.time() - start_time) < deadline

        # --- Phase 1: Structured exploration with Hammersley points ---
        # Hammersley points give low-discrepancy coverage of the box.
        # Use ~40% of budget for exploration.
        n_explore = min(budget // 2, 200)
        n_explore = max(n_explore, 50)

        # Generate Hammersley points in [0,1]^n, map to [-5,5]^n
        def hammersley_points(count, dim, rng):
            """Generate low-discrepancy points using radical inverse (Hammersley)."""
            points = np.zeros((count, dim))
            for j in range(dim):
                beta = j + 1  # base for radical inverse
                # Van der Corput sequence
                seq = np.zeros(count)
                for i in range(count):
                    f = 1.0
                    val = 0.0
                    ii = i + 1
                    while ii > 0:
                        f /= beta
                        val += f * (ii % beta)
                        ii //= beta
                    seq[i] = val
                points[:, j] = seq
            return points

        # Also add some random points for diversity
        hammersley = hammersley_points(n_explore, n, rng)
        # Map to [-5, 5]
        hammersley = hammersley * 10.0 - 5.0

        # Add a few random points and center
        n_rand = max(5, n_explore // 10)
        rand_pts = rng.uniform(-5, 5, (n_rand, n))
        center = np.zeros(n)
        all_explore = np.vstack([hammersley, rand_pts, center[None, :]])
        # Shuffle to interleave
        idx = rng.permutation(len(all_explore))
        all_explore = all_explore[idx]

        best_x = None
        best_f = np.inf
        call_count = 0

        for pt in all_explore:
            if not elapsed_ok():
                break
            if call_count >= budget:
                break
            x_try = np.clip(pt, -5, 5)
            f_try = func(x_try)
            call_count += 1
            if f_try < best_f:
                best_f = f_try
                best_x = x_try.copy()

        if best_x is None:
            # Fallback: just evaluate center
            best_x = np.zeros(n)
            best_f = func(best_x)
            call_count += 1

        # --- Phase 2: Adaptive (1+1)-ES with coordinate descent polish ---
        remaining = budget - call_count
        if remaining <= 0:
            return best_x, best_f

        x = best_x.copy()
        fx = best_f
        sigma = 1.0  # initial step size

        # Track stall for potential restarts
        stall_count = 0
        best_phase2_f = fx
        best_phase2_x = x.copy()
        restarts = 0
        max_restarts = 3

        # Coordinate descent polish function
        def coord_descent(x_cur, f_cur, max_steps, step_sizes):
            """Perform coordinate descent with adaptive step sizes."""
            x_cd = x_cur.copy()
            f_cd = f_cur
            for _ in range(max_steps):
                if not elapsed_ok():
                    break
                improved = False
                for d in range(n):
                    for sign in [1, -1]:
                        for s in step_sizes:
                            x_try = x_cd.copy()
                            x_try[d] += sign * s
                            x_try = np.clip(x_try, -5, 5)
                            f_try = func(x_try)
                            call_count_nonlocal = call_count  # we'll track outside
                            if f_try < f_cd:
                                x_cd = x_try
                                f_cd = f_try
                                improved = True
                                break
                    if improved:
                        break
                if not improved:
                    break
            return x_cd, f_cd

        # Main ES loop
        es_iter = 0
        while call_count < budget and elapsed_ok():
            # Try a few step sizes (adaptive)
            y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)
            call_count += 1

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.3
                stall_count = 0
                if fx < best_phase2_f:
                    best_phase2_f = fx
                    best_phase2_x = x.copy()
            else:
                sigma *= 0.85
                stall_count += 1

            # Periodic coordinate polish every 30 steps
            if es_iter % 30 == 29 and call_count < budget - 20:
                # Quick coordinate polish
                step_sizes = [0.5, 0.1, 0.01]
                x_polished, f_polished = coord_descent(x, fx, 3, step_sizes)
                if f_polished < fx:
                    x, fx = x_polished, f_polished
                    if fx < best_phase2_f:
                        best_phase2_f = fx
                        best_phase2_x = x.copy()
                    call_count += 3 * n  # approximate; actual calls tracked by func

            # Restart if stalled too much
            if stall_count > 25 and restarts < max_restarts:
                # Restart from a random point near the best
                x = best_phase2_x + 0.5 * rng.standard_normal(n)
                x = np.clip(x, -5, 5)
                fx = func(x)
                call_count += 1
                sigma = 1.0
                stall_count = 0
                restarts += 1

            es_iter += 1

        # --- Phase 3: Final high-precision coordinate polish ---
        remaining = budget - call_count
        if remaining > 10 and elapsed_ok():
            # Use the best point found
            x_final = best_phase2_x.copy()
            f_final = best_phase2_f

            # Fine coordinate descent with shrinking steps
            step_schedules = [
                [0.1, 0.01, 0.001],
                [0.01, 0.001, 0.0001],
                [0.001, 0.0001, 0.00001],
            ]
            for steps in step_schedules:
                if not elapsed_ok() or call_count >= budget - 5:
                    break
                for _ in range(min(5, (budget - call_count) // (2 * n))):
                    if not elapsed_ok() or call_count >= budget - 5:
                        break
                    for d in range(n):
                        if not elapsed_ok() or call_count >= budget - 5:
                            break
                        for sign in [1, -1]:
                            if not elapsed_ok() or call_count >= budget - 5:
                                break
                            x_try = x_final.copy()
                            x_try[d] += sign * steps[0]
                            x_try = np.clip(x_try, -5, 5)
                            f_try = func(x_try)
                            call_count += 1
                            if f_try < f_final:
                                x_final = x_try
                                f_final = f_try

        # Ensure we have a valid result
        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            call_count += 1

        # Return the best found
        if best_phase2_f < best_f:
            return best_phase2_x, best_phase2_f
        return best_x, best_f
# EVOLVE-BLOCK-END
