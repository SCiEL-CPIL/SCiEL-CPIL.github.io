"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead for BBOB f6 with diverse initializations."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0

        rng = np.random.default_rng(12345)

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Multi-start strategy: use diverse starting points
        # Phase 1: Coarse random sampling to find promising regions
        n_coarse = min(80, budget // 5)
        coarse_points = []

        # Always include origin
        x0 = np.zeros(dim)
        f0 = evaluate(x0)
        coarse_points.append((f0, x0))

        # Add random points spread across the domain
        for _ in range(n_coarse - 1):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, size=dim)
            f = evaluate(x)
            coarse_points.append((f, x))

        # Sort by function value
        coarse_points.sort(key=lambda t: t[0])

        # Phase 2: Local refinement with Nelder-Mead from top candidates
        # Use remaining budget for local searches
        remaining = budget - calls
        n_local = min(5, max(1, remaining // 50))
        local_budget_per = max(10, remaining // n_local)

        for i in range(min(n_local, len(coarse_points))):
            if calls >= budget or time.time() > deadline:
                break

            start_f, start_x = coarse_points[i]
            if start_f >= best_f:
                break

            # Nelder-Mead simplex method
            simplex = [start_x.copy()]
            simplex_f = [start_f]

            # Initialize simplex around start point
            scale = 0.5
            for j in range(dim):
                x_j = start_x.copy()
                x_j[j] += scale
                if x_j[j] > 5:
                    x_j[j] = start_x[j] - scale
                f_j = evaluate(x_j)
                simplex.append(x_j)
                simplex_f.append(f_j)

            # Nelder-Mead iterations
            alpha, gamma, rho, sigma_nm = 1.0, 2.0, 0.5, 0.5
            max_iter = local_budget_per
            iter_count = 0

            while iter_count < max_iter and calls < budget:
                if time.time() > deadline:
                    break
                iter_count += 1

                # Sort simplex by function value
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = evaluate(x_r)

                if f_r < simplex_f[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = evaluate(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                elif f_r < simplex_f[-2]:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    f_c = evaluate(x_c)
                    if f_c < simplex_f[-1]:
                        simplex[-1] = x_c
                        simplex_f[-1] = f_c
                    else:
                        # Shrink
                        for k in range(1, len(simplex)):
                            simplex[k] = (simplex[0] + simplex[k]) / 2.0
                            simplex_f[k] = evaluate(simplex[k])

            # Update best from local search
            if min(simplex_f) < best_f:
                best_idx = np.argmin(simplex_f)
                best_x = simplex[best_idx].copy()
                best_f = simplex_f[best_idx]

        return best_x, best_f
# EVOLVE-BLOCK-END
