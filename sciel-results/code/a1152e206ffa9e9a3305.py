"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with Sobol sampling + CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit

        # Call counter
        calls = [0]

        def safe_func(x):
            if calls[0] >= budget:
                raise StopIteration
            calls[0] += 1
            return float(func(np.asarray(x, dtype=np.float64)))

        # Sobol-like quasi-random points for uniform coverage
        # Use a simple Halton sequence as Sobol substitute (standard lib + numpy only)
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        def halton_sequence(n, dim):
            points = np.zeros((n, dim))
            for i in range(n):
                for d in range(dim):
                    base = d + 2  # 2, 3, 5, 7, 11...
                    points[i, d] = halton(i + 1, base)
            return points

        # Phase 1: Multi-start Nelder-Mead with Sobol/Halton sampling
        # Allocate ~60% of budget to multi-start local searches
        num_starts = 50
        budget_per_start = max(5, (int(budget * 0.6) - num_starts) // num_starts)

        best_x_global = None
        best_f_global = np.inf

        # Generate Halton points in [-5, 5]^dim
        halton_pts = halton_sequence(num_starts, dim)
        # Scale to [-5, 5]
        halton_pts = -5.0 + 10.0 * halton_pts

        for s in range(num_starts):
            if calls[0] >= budget or time.time() > deadline:
                break

            x0 = np.clip(halton_pts[s], -5, 5)
            try:
                f0 = safe_func(x0)
            except StopIteration:
                break

            if f0 < best_f_global:
                best_f_global = f0
                best_x_global = x0.copy()

            # Nelder-Mead local search
            x = x0.copy()
            f = f0

            # Initial simplex: x plus small perturbations along each axis
            simplex = [x.copy()]
            step = 0.5
            for d in range(dim):
                v = x.copy()
                v[d] += step
                v = np.clip(v, -5, 5)
                simplex.append(v)

            # Evaluate simplex
            simplex_f = [f]
            for i in range(1, dim + 1):
                if calls[0] >= budget or time.time() > deadline:
                    break
                try:
                    simplex_f.append(safe_func(simplex[i]))
                except StopIteration:
                    break

            # Nelder-Mead iterations
            alpha, gamma, rho = 1.0, 2.0, 0.5
            max_iters = budget_per_start
            for it in range(max_iters):
                if calls[0] >= budget or time.time() > deadline:
                    break
                if len(simplex_f) < dim + 1:
                    break

                # Sort simplex by function value
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                try:
                    fr = safe_func(xr)
                except StopIteration:
                    break

                if fr < simplex_f[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    try:
                        fe = safe_func(xe)
                    except StopIteration:
                        break
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    # Contraction
                    xc = centroid + rho * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    try:
                        fc = safe_func(xc)
                    except StopIteration:
                        break
                    if fc < simplex_f[-1]:
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            try:
                                simplex_f[i] = safe_func(simplex[i])
                            except StopIteration:
                                break
                        if calls[0] >= budget or time.time() > deadline:
                            break

                # Update global best
                min_idx = np.argmin(simplex_f)
                if simplex_f[min_idx] < best_f_global:
                    best_f_global = simplex_f[min_idx]
                    best_x_global = simplex[min_idx].copy()

                # Convergence check
                f_range = max(simplex_f) - min(simplex_f)
                if f_range < 1e-10:
                    break

        # Phase 2: CMA-ES-style refinement at global best
        if calls[0] < budget and best_x_global is not None:
            center = best_x_global.copy()
            cf = best_f_global
            sigma = 0.3
            rng = np.random.default_rng(123)

            while calls[0] < budget and time.time() < deadline:
                improved_any = False
                lam = 4
                for i in range(lam):
                    if calls[0] >= budget or time.time() > deadline:
                        break
                    child = np.clip(center + sigma * rng.standard_normal(dim), -5, 5)
                    try:
                        cf_c = safe_func(child)
                    except StopIteration:
                        break
                    if cf_c < cf:
                        cf = cf_c
                        center = child.copy()
                        improved_any = True

                # Adapt sigma
                if improved_any:
                    sigma *= 1.15
                else:
                    sigma *= 0.85

                if sigma < 1e-8:
                    break

            if cf < best_f_global:
                best_f_global = cf
                best_x_global = center

        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
