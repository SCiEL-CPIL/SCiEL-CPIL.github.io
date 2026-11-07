"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start local search: global sampling + Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        t_start = time.time()
        deadline = t_start + 280.0

        dim = self.dim
        budget = self.budget
        calls = 0

        def clip(x):
            return np.clip(x, -5.0, 5.0)

        # Phase 1: global exploration - sample points
        # Use a mix of center-biased and spread-out points
        n_global = min(30, budget // 10)
        points = []
        # Center point
        points.append(np.zeros(dim))
        # Random points in [-4, 4] since minimizer is there
        for _ in range(n_global - 1):
            points.append(rng.uniform(-4, 4, dim))

        best_x = None
        best_f = np.inf

        for p in points:
            if calls >= budget or time.time() > deadline:
                break
            p = clip(p)
            f = func(p)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = p.copy()

        # Phase 2: Nelder-Mead local refinement from best seed
        # Implement a fast simplex method
        def nelder_mead(x0, f0, max_iter=50):
            nonlocal calls, best_f, best_x
            n = dim
            # Build simplex
            simplex = [x0.copy()]
            fvals = [f0]
            for i in range(n):
                xi = x0.copy()
                xi[i] += 0.5 if xi[i] < 4.5 else -0.5
                xi = clip(xi)
                fi = func(xi)
                calls += 1
                if calls >= budget or time.time() > deadline:
                    return
                simplex.append(xi)
                fvals.append(fi)

            for _ in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    return
                # Sort by f
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                f_worst = fvals[-1]

                # Reflection
                alpha = 1.0
                xr = clip(centroid + alpha * (centroid - worst))
                fr = func(xr)
                calls += 1
                if calls >= budget or time.time() > deadline:
                    return

                if fr < fvals[0]:
                    # Expansion
                    gamma = 2.0
                    xe = clip(centroid + gamma * (centroid - worst))
                    fe = func(xe)
                    calls += 1
                    if calls >= budget or time.time() > deadline:
                        return
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    rho = 0.5
                    xc = clip(centroid + rho * (worst - centroid))
                    fc = func(xc)
                    calls += 1
                    if calls >= budget or time.time() > deadline:
                        return
                    if fc < f_worst:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]))
                            fvals[i] = func(simplex[i])
                            calls += 1
                            if calls >= budget or time.time() > deadline:
                                return

                # Track best
                for i in range(n + 1):
                    if fvals[i] < best_f:
                        best_f = fvals[i]
                        best_x = simplex[i].copy()

        # Refine from best seed found in Phase 1
        if best_x is not None and calls < budget and time.time() < deadline:
            nelder_mead(best_x, best_f, max_iter=80)

        # If we have budget left, try a few more local refinements from other good points
        # (We already tracked best, but let's do one more pass if time permits)
        if calls < budget and time.time() < deadline:
            # Try a second Nelder-Mead from a slightly perturbed best
            x2 = clip(best_x + 0.3 * rng.standard_normal(dim))
            f2 = func(x2)
            calls += 1
            if f2 < best_f:
                best_f = f2
                best_x = x2.copy()
            if calls < budget and time.time() < deadline:
                nelder_mead(best_x, best_f, max_iter=40)

        return best_x, best_f
# EVOLVE-BLOCK-END
