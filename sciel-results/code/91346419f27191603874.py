"""Combined Nelder-Mead simplex with adaptive local search for BBOB-style f6."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex with multi-start, refined by adaptive local search."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # 15s safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def time_ok():
            return time.time() < deadline and calls_made < budget - 2

        # Multi-start Nelder-Mead simplex: try several different starting simplices
        n_starts = 4
        start_seeds = [100, 200, 300, 400]
        start_scales = [2.0, 3.0, 4.0, 1.5]

        for start_idx in range(n_starts):
            if calls_made >= budget - 10:
                break
            if time.time() > deadline:
                break

            rng = np.random.default_rng(start_seeds[start_idx])
            scale = start_scales[start_idx]

            # Create initial simplex: one point at center, others offset along axes
            center = rng.uniform(-3.0, 3.0, dim)
            simplex = []
            fvals = []

            # First vertex: center
            v0 = np.clip(center, -5, 5)
            f0 = func(v0)
            calls_made += 1
            simplex.append(v0)
            fvals.append(f0)

            if f0 < best_f:
                best_f = f0
                best_x = v0.copy()

            # Additional vertices: offset along each axis
            for i in range(dim):
                vi = v0.copy()
                vi[i] += scale
                vi = np.clip(vi, -5, 5)
                fi = func(vi)
                calls_made += 1
                simplex.append(vi)
                fvals.append(fi)

                if fi < best_f:
                    best_f = fi
                    best_x = vi.copy()

                if calls_made >= budget - 10:
                    break

            if len(simplex) < dim + 1:
                break

            simplex = np.array(simplex[:dim + 1])
            fvals = np.array(fvals[:dim + 1])

            # Nelder-Mead parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma_nm = 0.5  # shrink

            # Run Nelder-Mead iterations
            max_iters = 100
            for iteration in range(max_iters):
                if calls_made >= budget - 5:
                    break
                if time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                # Check convergence
                if np.max(fvals) - np.min(fvals) < 1e-10:
                    break

                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Worst point
                worst = simplex[-1]
                worst_f = fvals[-1]

                # Reflection
                xr = centroid + alpha * (centroid - worst)
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls_made += 1

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - worst)
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls_made += 1

                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Simple reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if fr < worst_f:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                        xc = np.clip(xc, -5, 5)
                        fc = func(xc)
                        calls_made += 1
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - worst)
                        xc = np.clip(xc, -5, 5)
                        fc = func(xc)
                        calls_made += 1

                    if fc < min(fr, worst_f):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fi = func(simplex[i])
                            calls_made += 1
                            fvals[i] = fi

                # Update global best
                min_idx = np.argmin(fvals)
                if fvals[min_idx] < best_f:
                    best_f = fvals[min_idx]
                    best_x = simplex[min_idx].copy()

                # Early exit if simplex is very small
                if np.linalg.norm(simplex[0] - simplex[-1]) < 1e-8:
                    break

        # Final polish: adaptive step-size local search around the best point found
        if best_x is not None and calls_made < budget:
            if time.time() < deadline:
                rng = np.random.default_rng(999)
                x = best_x.copy()
                fx = best_f
                sigma = max(1e-3, 0.1)
                sigma_min = 1e-6
                sigma_max = 1.0
                success = 0
                fail = 0

                for _ in range(min(100, budget - calls_made)):
                    if time.time() > deadline:
                        break
                    if calls_made >= budget - 1:
                        break

                    direction = rng.standard_normal(dim)
                    y = np.clip(x + sigma * direction, -5, 5)
                    fy = func(y)
                    calls_made += 1

                    if fy < fx:
                        x, fx = y, fy
                        success += 1
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                        sigma = min(sigma * 1.15, sigma_max)
                    else:
                        fail += 1
                        sigma = max(sigma * 0.75, sigma_min)

                    total = success + fail
                    if total > 40:
                        success = 0
                        fail = 0

                    if sigma < sigma_min * 10 and total > 20:
                        break

        # Ensure we have a valid result
        if best_x is None:
            x = np.zeros(dim)
            best_x = x
            if calls_made < budget:
                best_f = func(x)
                calls_made += 1
            else:
                best_f = 0.0

        return best_x, best_f
# EVOLVE-BLOCK-END
