"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with Rosenbrock-aware initialization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Rosenbrock valley: f(x) = sum 100*(x_{i+1}-x_i^2)^2 + (1-x_i)^2
        # The valley is along x_{i+1} = x_i^2, with minimum at x_i = 1 for all i
        # But the actual minimum is shifted/rotated/offset by hidden transforms.
        # Strategy: multi-start Nelder-Mead with diverse initial points.

        rng = np.random.default_rng(42)

        best_x = None
        best_f = float('inf')
        calls_used = 0

        def safe_func(x):
            nonlocal calls_used
            x = np.asarray(x, dtype=np.float64).flatten()
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            calls_used += 1
            return val

        # Generate diverse starting points
        # Mix of: near origin, random, and Rosenbrock-valley-inspired points
        starts = []

        # A few points near the origin and various locations
        for _ in range(10):
            starts.append(rng.uniform(-3, 3, dim))

        # Points along the Rosenbrock valley pattern (shifted)
        for shift in [-2, -1, 0, 1, 2]:
            pt = np.ones(dim) + shift * 0.5
            pt = pt + rng.normal(0, 0.5, dim)
            starts.append(np.clip(pt, -5, 5))

        # Pure random
        for _ in range(5):
            starts.append(rng.uniform(-5, 5, dim))

        # Shuffle starts
        order = rng.permutation(len(starts))
        starts = [starts[i] for i in order]

        # For each start, run Nelder-Mead with a sub-budget
        n_starts = len(starts)
        # Reserve some budget for final polish
        per_start_budget = max(20, (budget - dim) // n_starts)

        for idx, x0 in enumerate(starts):
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break

            x0 = np.clip(x0, -5, 5)
            fx0 = safe_func(x0)
            if fx0 < best_f:
                best_f = fx0
                best_x = x0.copy()

            # Build initial simplex
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                step = 0.5 if abs(x0[i]) < 1e-8 else 0.05 * abs(x0[i])
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                simplex.append(xi)

            # Evaluate simplex
            simplex_f = [fx0]
            for i in range(1, dim + 1):
                if calls_used >= budget or time.time() > deadline:
                    break
                simplex_f.append(safe_func(simplex[i]))

            if len(simplex_f) < dim + 1:
                continue

            # Nelder-Mead iterations
            nm_budget = per_start_budget - (dim + 1)
            alpha, gamma, rho, sigma = 1.0, 2.0, -0.5, 0.5

            for _ in range(nm_budget):
                if calls_used >= budget or time.time() > deadline:
                    break

                # Sort simplex by function value
                idxs = np.argsort(simplex_f)
                simplex = [simplex[i] for i in idxs]
                simplex_f = [simplex_f[i] for i in idxs]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fxr = safe_func(xr)

                if fxr < simplex_f[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fxe = safe_func(xe)
                    if fxe < fxr:
                        simplex[-1] = xe
                        simplex_f[-1] = fxe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fxr
                elif fxr < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    simplex_f[-1] = fxr
                else:
                    # Contraction
                    if fxr < simplex_f[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    fxc = safe_func(xc)
                    if fxc < min(fxr, simplex_f[-1]):
                        simplex[-1] = xc
                        simplex_f[-1] = fxc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = safe_func(simplex[i])

                # Track best
                if simplex_f[-1] < best_f:
                    best_f = simplex_f[-1]
                    best_x = simplex[-1].copy()

        # Final polish: a few local steps from best
        if best_x is not None and calls_used < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            step_size = 0.1
            for _ in range(min(50, budget - calls_used)):
                if time.time() > deadline:
                    break
                improved = False
                for i in range(dim):
                    for direction in [1, -1]:
                        y = x.copy()
                        y[i] += direction * step_size
                        y = np.clip(y, -5, 5)
                        fy = safe_func(y)
                        if fy < fx:
                            x, fx = y, fy
                            improved = True
                if not improved:
                    step_size *= 0.7
                if step_size < 1e-6:
                    break
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
