"""Sequential global-to-local Nelder-Mead optimizer for BBOB f20 (Schwefel).

Strategy:
  1. Global phase: large-scale multi-start Nelder-Mead with coarse initial
     simplex to rapidly traverse the multimodal landscape.
  2. Local phase: progressively shrink the simplex scale around the best
     point found for high-precision convergence.
  3. Strict budget and deadline guards.
"""
import numpy as np
import time


class Optimizer:
    """Sequential global-to-local Nelder-Mead optimizer for BBOB f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 295.0

        call_count = [0]

        def make_call(x):
            if call_count[0] >= budget:
                return None
            call_count[0] += 1
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            return float(func(x))

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def time_ok():
            return time.time() < deadline - 3.0

        def budget_ok():
            return call_count[0] < budget - 2

        def nelder_mead(start_x, step_scale, max_iter):
            """Run Nelder-Mead from start_x with given initial step scale."""
            nonlocal best_x, best_f

            # Build initial simplex
            simplex = [start_x.copy()]
            fvals = []
            f0 = make_call(start_x)
            if f0 is None:
                return
            update_best(start_x, f0)
            fvals.append(f0)

            for i in range(dim):
                xi = start_x.copy()
                xi[i] += step_scale
                xi = np.clip(xi, -5.0, 5.0)
                fi = make_call(xi)
                if fi is None:
                    return
                update_best(xi, fi)
                simplex.append(xi)
                fvals.append(fi)

            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

            for _ in range(max_iter):
                if not budget_ok() or not time_ok():
                    break

                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]

                if abs(fvals[0] - fvals[-1]) < 1e-12:
                    break

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                fr = make_call(xr)
                if fr is None:
                    break
                update_best(xr, fr)

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (xr - centroid)
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = make_call(xe)
                    if fe is None:
                        break
                    update_best(xe, fe)
                    if fe < fr:
                        simplex[-1], fvals[-1] = xe, fe
                    else:
                        simplex[-1], fvals[-1] = xr, fr
                elif fr < fvals[-2]:
                    simplex[-1], fvals[-1] = xr, fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        xc = centroid + 0.5 * (xr - centroid)
                    else:
                        xc = centroid - 0.5 * (centroid - simplex[-1])
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = make_call(xc)
                    if fc is None:
                        break
                    update_best(xc, fc)
                    if fc < fr:
                        simplex[-1], fvals[-1] = xc, fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fs = make_call(simplex[i])
                            if fs is None:
                                break
                            update_best(simplex[i], fs)
                            fvals[i] = fs
                        if not budget_ok() or not time_ok():
                            break

        # Phase 1: Global search with large-scale multi-start
        n_global = 12
        global_scales = [3.0, 2.5, 2.0, 1.5, 1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2]

        for i in range(n_global):
            if not budget_ok() or not time_ok():
                break

            scale = global_scales[i % len(global_scales)]
            # Diverse starting points
            if i == 0:
                start = np.zeros(dim)
            elif i == 1:
                start = np.full(dim, 4.20968)
            else:
                start = rng.uniform(-5.0, 5.0, dim)

            # Max iterations based on remaining budget
            remaining = budget - call_count[0]
            max_iter = max(5, min(40, int(remaining / 15)))

            nelder_mead(start, scale, max_iter)

        # Phase 2: Local refinement with progressively smaller scales
        if best_x is not None:
            local_scales = [0.15, 0.1, 0.07, 0.05, 0.03, 0.02, 0.01, 0.005]
            for scale in local_scales:
                if not budget_ok() or not time_ok():
                    break
                if best_x is None:
                    break

                # Start from best point with small perturbation
                start = best_x.copy()
                start += rng.normal(0, scale * 0.5, dim)
                start = np.clip(start, -5.0, 5.0)

                remaining = budget - call_count[0]
                max_iter = max(5, min(50, int(remaining / 10)))

                nelder_mead(start, scale, max_iter)

        # Phase 3: Final ultra-fine polish if budget remains
        if best_x is not None and budget_ok() and time_ok():
            for scale in [0.003, 0.002, 0.001, 0.0005, 0.0003]:
                if not budget_ok() or not time_ok():
                    break
                start = best_x.copy()
                start += rng.normal(0, scale * 0.3, dim)
                start = np.clip(start, -5.0, 5.0)

                remaining = budget - call_count[0]
                max_iter = max(3, min(30, int(remaining / 8)))

                nelder_mead(start, scale, max_iter)

        return best_x, best_f
