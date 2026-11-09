"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with periodic exploration and final Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # Safety margin before 330s kill

        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        # Initial evaluation
        x_best = rng.uniform(-4, 4, n)
        fx_best = func(x_best)
        calls = 1

        # Parameters
        sigma = 1.5
        sigma_min = 1e-5
        sigma_max = 8.0
        stall_limit = 40
        stall_count = 0
        window_size = 12
        successes = []
        explore_interval = 35
        calls_since_explore = 0

        # Track best for final return
        x_nm_best = x_best.copy()
        fx_nm_best = fx_best

        # Main loop: (1+1)-ES with periodic exploration
        while calls < budget - 15 and time.time() < deadline:
            calls_since_explore += 1

            # Periodic exploration: jump to a new random point
            if calls_since_explore >= explore_interval:
                calls_since_explore = 0
                x_jump = rng.uniform(-5, 5, n)
                fx_jump = func(x_jump)
                calls += 1
                if fx_jump < fx_best:
                    x_best = x_jump
                    fx_best = fx_jump
                    if fx_best < fx_nm_best:
                        x_nm_best = x_best.copy()
                        fx_nm_best = fx_best
                    sigma = min(sigma * 1.3, sigma_max)
                    stall_count = 0
                continue

            # Generate candidate
            y = np.clip(x_best + sigma * rng.standard_normal(n), -5, 5)
            fy = func(y)
            calls += 1

            if fy < fx_best:
                x_best = y
                fx_best = fy
                if fx_best < fx_nm_best:
                    x_nm_best = x_best.copy()
                    fx_nm_best = fx_best
                successes.append(1)
                stall_count = 0
            else:
                successes.append(0)
                stall_count += 1

            # Update window
            if len(successes) > window_size:
                successes.pop(0)

            # Adapt sigma based on success rate
            if len(successes) >= window_size:
                success_rate = np.mean(successes)
                if success_rate > 0.25:
                    sigma = min(sigma * 1.12, sigma_max)
                elif success_rate < 0.15:
                    sigma = max(sigma * 0.88, sigma_min)

            # Stall detection: restart from new random point
            if stall_count >= stall_limit:
                stall_count = 0
                sigma = 1.5
                x_new = rng.uniform(-5, 5, n)
                fx_new = func(x_new)
                calls += 1
                if fx_new < fx_best:
                    x_best = x_new
                    fx_best = fx_new
                    if fx_best < fx_nm_best:
                        x_nm_best = x_best.copy()
                        fx_nm_best = fx_best
                successes = []

        # Final Nelder-Mead polish with remaining budget
        if calls < budget - 1 and time.time() < deadline:
            x0 = x_nm_best.copy()
            simplex = [x0.copy()]
            for i in range(n):
                xi = x0.copy()
                xi[i] += 0.1
                simplex.append(xi)

            vals = []
            for i in range(n + 1):
                if calls >= budget or time.time() > deadline:
                    break
                v = func(simplex[i])
                vals.append(v)
                calls += 1
                if v < fx_nm_best:
                    fx_nm_best = v
                    x_nm_best = simplex[i].copy()

            # Run NM iterations
            max_nm_iters = 20
            for _ in range(max_nm_iters):
                if calls >= budget or time.time() > deadline:
                    break
                if len(vals) != n + 1:
                    break

                # Sort simplex
                idx = np.argsort(vals)
                simplex = [simplex[i] for i in idx]
                vals = [vals[i] for i in idx]

                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                fw = vals[-1]

                # Reflect
                xr = centroid + 1.0 * (centroid - worst)
                xr = np.clip(xr, -5, 5)
                fxr = func(xr)
                calls += 1
                if fxr < fx_nm_best:
                    fx_nm_best = fxr
                    x_nm_best = xr.copy()

                if fxr < vals[0]:
                    # Expand
                    xe = centroid + 2.0 * (centroid - worst)
                    xe = np.clip(xe, -5, 5)
                    fxe = func(xe)
                    calls += 1
                    if fxe < fx_nm_best:
                        fx_nm_best = fxe
                        x_nm_best = xe.copy()
                    if fxe < fxr:
                        simplex[-1] = xe
                        vals[-1] = fxe
                    else:
                        simplex[-1] = xr
                        vals[-1] = fxr
                elif fxr < vals[-2]:
                    simplex[-1] = xr
                    vals[-1] = fxr
                else:
                    # Contract
                    xc = centroid + 0.5 * (worst - centroid)
                    xc = np.clip(xc, -5, 5)
                    fxc = func(xc)
                    calls += 1
                    if fxc < fx_nm_best:
                        fx_nm_best = fxc
                        x_nm_best = xc.copy()
                    if fxc < fw:
                        simplex[-1] = xc
                        vals[-1] = fxc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if calls < budget:
                                vals[i] = func(simplex[i])
                                calls += 1
                                if vals[i] < fx_nm_best:
                                    fx_nm_best = vals[i]
                                    x_nm_best = simplex[i].copy()

        return x_nm_best, fx_nm_best
# EVOLVE-BLOCK-END
