"""Optimizer for rotated Rosenbrock (BBOB f9) using Nelder-Mead simplex with basin identification."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Nelder-Mead simplex with sparse global lattice for rotated Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        call_count = 0
        best_x = None
        best_fx = np.inf

        def tracked_func(x):
            nonlocal call_count, best_x, best_fx
            if time.time() > deadline:
                return np.inf
            fx = func(np.clip(x, -5, 5))
            call_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # Phase 1: Sparse global lattice to identify basin
        # Use a stratified sampling approach for better coverage
        n_grid = 3  # 3^5 = 243 points, but we'll use a subset
        grid_points = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    for l in range(2):
                        for m in range(2):
                            grid_points.append([
                                -3.0 + 6.0 * i,
                                -3.0 + 6.0 * j,
                                -3.0 + 6.0 * k,
                                -3.0 + 6.0 * l,
                                -3.0 + 6.0 * m
                            ])
        # Also add center and some random points
        grid_points.append([0.0] * dim)
        grid_points.append([1.0] * dim)
        grid_points.append([-1.0] * dim)
        grid_points.append([2.0] * dim)
        grid_points.append([-2.0] * dim)

        # Shuffle and evaluate a subset
        rng.shuffle(grid_points)
        n_eval = min(40, len(grid_points), budget // 3)
        candidates = []
        for pt in grid_points[:n_eval]:
            if not time_ok() or call_count >= budget:
                break
            pt = np.array(pt, dtype=float)
            fx = tracked_func(pt)
            if fx < np.inf:
                candidates.append((fx, pt))

        if not candidates:
            # Fallback: random initialization
            x0 = rng.uniform(-4, 4, dim)
            fx = tracked_func(x0)
            candidates.append((fx, x0))

        # Sort by function value
        candidates.sort(key=lambda t: t[0])
        start_x = candidates[0][1].copy()

        # Phase 2: Nelder-Mead simplex method
        def nelder_mead(x0, max_iter=200):
            """Nelder-Mead simplex optimization."""
            nonlocal call_count
            n = dim
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink

            # Initialize simplex
            simplex = [x0.copy()]
            for i in range(n):
                dx = x0.copy()
                step = 0.1 if abs(x0[i]) > 0.01 else 0.001
                dx[i] += step
                simplex.append(dx)

            # Evaluate initial simplex
            fvals = []
            for i in range(n + 1):
                if not time_ok() or call_count >= budget:
                    break
                fx = tracked_func(simplex[i])
                fvals.append(fx)
            if len(fvals) < n + 1:
                return None

            # Main iteration loop
            for _ in range(max_iter):
                if not time_ok() or call_count >= budget:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                fx_r = tracked_func(x_r)

                if fx_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    fx_e = tracked_func(x_e)
                    if fx_e < fx_r:
                        simplex[-1] = x_e
                        fvals[-1] = fx_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = fx_r
                elif fx_r < fvals[-2]:
                    # Reflection accepted
                    simplex[-1] = x_r
                    fvals[-1] = fx_r
                else:
                    # Contraction
                    if fx_r < fvals[-1]:
                        # Outside contraction
                        x_c = centroid + rho * (x_r - centroid)
                        fx_c = tracked_func(x_c)
                        if fx_c <= fx_r:
                            simplex[-1] = x_c
                            fvals[-1] = fx_c
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                if not time_ok() or call_count >= budget:
                                    break
                                fvals[i] = tracked_func(simplex[i])
                            break
                    else:
                        # Inside contraction
                        x_c = centroid - rho * (centroid - simplex[-1])
                        fx_c = tracked_func(x_c)
                        if fx_c <= fvals[-1]:
                            simplex[-1] = x_c
                            fvals[-1] = fx_c
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                if not time_ok() or call_count >= budget:
                                    break
                                fvals[i] = tracked_func(simplex[i])
                            break

                # Check convergence
                if len(fvals) == n + 1:
                    f_range = max(fvals) - min(fvals)
                    if f_range < 1e-10:
                        break

            # Return best point
            if simplex:
                order = np.argsort(fvals)
                return simplex[order[0]]
            return None

        # Run Nelder-Mead from the best basin point
        if time_ok() and call_count < budget:
            result = nelder_mead(start_x, max_iter=150)
            if result is not None:
                start_x = result

        # Phase 3: If budget remains, do a second round of Nelder-Mead from other good basins
        if time_ok() and call_count < budget - 50:
            for fx, pt in candidates[1:4]:
                if not time_ok() or call_count >= budget:
                    break
                # Only try if this basin is reasonably close to the best
                if fx < best_fx + 50:
                    result = nelder_mead(pt, max_iter=100)
                    if result is not None:
                        start_x = result
                        if not time_ok() or call_count >= budget:
                            break

        # Final phase: fine-tuning with small Nelder-Mead
        if time_ok() and call_count < budget - 20:
            result = nelder_mead(start_x, max_iter=100)
            if result is not None:
                start_x = result

        return best_x, best_fx
# EVOLVE-BLOCK-END
