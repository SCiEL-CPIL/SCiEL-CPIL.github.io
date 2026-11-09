"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart (1+1)-ES with adaptive step size, time guard, and coordinate-descent refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        # Time management
        start_time = time.time()
        time_limit = 320.0  # 10s safety margin before 330s limit

        # RNG
        rng = np.random.default_rng(42)

        # Global best tracking
        x_best = np.zeros(n)
        f_best = np.inf
        eval_count = 0

        def try_eval(x):
            """Evaluate x, update global best, return (x, fx)."""
            nonlocal f_best, x_best, eval_count
            fx = func(x)
            eval_count += 1
            if fx < f_best:
                f_best = fx
                x_best = x.copy()
            return x, fx

        # Phase 1: Multi-start initialization
        # Evaluate center + 9 random points to find a good starting basin
        init_points = [np.zeros(n)]
        for _ in range(9):
            init_points.append(rng.uniform(-5, 5, n))

        for x_init in init_points:
            if time.time() - start_time > time_limit or eval_count >= budget:
                break
            try:
                x_init, fx_init = try_eval(x_init)
            except Exception:
                break

        # Phase 2: Main search loop with (1+1)-ES and coordinate-descent refinement
        # We use a multi-restart approach where each restart does:
        # 1. A (1+1)-ES local search with adaptive sigma
        # 2. A coordinate-descent refinement pass on the best point found

        sigma_min = 1e-10
        sigma_max = 8.0
        restart_threshold = 35  # Steps without global improvement before restart

        while eval_count < budget:
            if time.time() - start_time > time_limit:
                break

            # Initialize this restart from the global best
            x = x_best.copy()
            fx = f_best
            sigma = 1.5
            steps_since_best = 0

            # (1+1)-ES local search
            while eval_count < budget:
                if time.time() - start_time > time_limit:
                    break

                # Generate candidate
                z = rng.standard_normal(n)
                y = x + sigma * z
                y = np.clip(y, -5, 5)

                try:
                    fy = func(y)
                except Exception:
                    break
                eval_count += 1

                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.15
                    if fy < f_best:
                        f_best = fy
                        x_best = x.copy()
                        steps_since_best = 0
                    else:
                        steps_since_best += 1
                else:
                    sigma *= 0.88
                    steps_since_best += 1

                # Clamp sigma
                sigma = np.clip(sigma, sigma_min, sigma_max)

                # Check for stagnation
                if steps_since_best >= restart_threshold:
                    break

                # Early termination if sigma is very small
                if sigma < 0.02 and steps_since_best > 15:
                    break

            # Coordinate-descent refinement on the current best
            # This exploits axis-aligned structure for rapid convergence
            if eval_count < budget and time.time() - start_time < time_limit:
                x_cd = x_best.copy()
                fx_cd = f_best
                improved = True
                max_cd_passes = 3  # Limit CD passes to conserve budget

                for pass_num in range(max_cd_passes):
                    if not improved or eval_count >= budget:
                        break
                    if time.time() - start_time > time_limit:
                        break
                    improved = False

                    for d in range(n):
                        if eval_count >= budget or time.time() - start_time > time_limit:
                            break

                        # Binary search for optimal value of coordinate d
                        lo = -5.0
                        hi = 5.0
                        # Start with a coarse grid, then refine
                        for _ in range(6):  # 6 iterations of binary search
                            if eval_count >= budget or time.time() - start_time > time_limit:
                                break
                            mid = (lo + hi) / 2.0
                            # Evaluate at mid
                            x_test = x_cd.copy()
                            x_test[d] = mid
                            try:
                                f_mid = func(x_test)
                            except Exception:
                                break
                            eval_count += 1

                            # Also evaluate at a point slightly offset to determine direction
                            offset = (hi - lo) * 0.25
                            x_right = x_cd.copy()
                            x_right[d] = min(5.0, mid + offset)
                            try:
                                f_right = func(x_right)
                            except Exception:
                                break
                            eval_count += 1

                            if f_right < f_mid:
                                lo = mid
                            else:
                                hi = mid

                            # Update best if we found something better
                            if f_mid < fx_cd:
                                fx_cd = f_mid
                                x_cd = x_test.copy()
                                improved = True
                            if f_right < fx_cd:
                                fx_cd = f_right
                                x_cd = x_right.copy()
                                improved = True

                        # Final evaluation at the best found position
                        if eval_count < budget and time.time() - start_time < time_limit:
                            x_final = x_cd.copy()
                            x_final[d] = (lo + hi) / 2.0
                            try:
                                f_final = func(x_final)
                            except Exception:
                                break
                            eval_count += 1
                            if f_final < fx_cd:
                                fx_cd = f_final
                                x_cd = x_final.copy()
                                improved = True

                    # Update global best
                    if fx_cd < f_best:
                        f_best = fx_cd
                        x_best = x_cd.copy()

            # If no improvement after CD, restart from a new random point
            # to escape local minima
            if eval_count < budget and time.time() - start_time < time_limit:
                # Check if we've been stuck for a while
                # Generate a new random starting point for the next restart
                pass  # The while loop will re-initialize from x_best

        return x_best, f_best
# EVOLVE-BLOCK-END
