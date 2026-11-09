"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # 300s limit with 15s safety margin

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_fx = np.inf
        calls_made = 0

        def timed_out():
            return time.time() > deadline - 3.0

        def nelder_mead(x0, max_iters, sigma_init=0.5):
            """Nelder-Mead simplex method with adaptive step sizes."""
            nonlocal calls_made, best_x, best_fx

            # Build initial simplex
            simplex = []
            fvals = []

            # First vertex: x0
            simplex.append(x0.copy())
            fvals.append(func(x0))
            calls_made += 1
            if fvals[0] < best_fx:
                best_fx = fvals[0]
                best_x = x0.copy()

            # Additional vertices: perturb each dimension
            for d in range(dim):
                if calls_made >= budget or timed_out():
                    break
                v = x0.copy()
                v[d] += sigma_init
                v = np.clip(v, -5, 5)
                simplex.append(v)
                fv = func(v)
                fvals.append(fv)
                calls_made += 1
                if fv < best_fx:
                    best_fx = fv
                    best_x = v.copy()

            simplex = np.array(simplex)
            fvals = np.array(fvals)
            n = len(simplex)  # dim + 1

            # Nelder-Mead parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            for it in range(max_iters):
                if calls_made >= budget or timed_out():
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                # Check convergence
                if fvals[0] > fvals[-1] - 1e-10:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                calls_made += 1
                if fr < best_fx:
                    best_fx = fr
                    best_x = xr.copy()

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    calls_made += 1
                    if fe < best_fx:
                        best_fx = fe
                        best_x = xe.copy()
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls_made += 1
                    if fc < best_fx:
                        best_fx = fc
                        best_x = xc.copy()

                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fi = func(simplex[i])
                            calls_made += 1
                            if fi < best_fx:
                                best_fx = fi
                                best_x = simplex[i].copy()
                            fvals[i] = fi
                        if calls_made >= budget or timed_out():
                            break

            return best_fx

        # Phase 1: Diverse initial sampling (important for AOCC early progress)
        n_initial = 10
        for i in range(n_initial):
            if calls_made >= budget or timed_out():
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # Phase 2: Multi-start Nelder-Mead from diverse starting points
        n_starts = 5
        remaining_budget = budget - calls_made
        # Distribute budget among starts, leaving some for refinement
        iters_per_start = max(20, remaining_budget // (n_starts * 2))

        for s in range(n_starts):
            if calls_made >= budget or timed_out():
                break

            # Choose start point: mix of random and near-best
            if s == 0:
                x0 = best_x.copy()
            elif s == 1:
                x0 = best_x + 0.5 * rng.standard_normal(dim)
                x0 = np.clip(x0, -5, 5)
            else:
                x0 = rng.uniform(-5, 5, dim)

            # Reduce iterations for later starts if budget is tight
            iters = min(iters_per_start, max(10, (budget - calls_made) // max(1, n_starts - s)))
            nelder_mead(x0, max_iters=iters)

        # Phase 3: Final refinement from best point if budget remains
        if calls_made < budget and not timed_out() and best_x is not None:
            remaining = budget - calls_made
            if remaining > 50:
                nelder_mead(best_x.copy(), max_iters=min(200, remaining // 2), sigma_init=0.1)

        return best_x, best_fx
# EVOLVE-BLOCK-END
