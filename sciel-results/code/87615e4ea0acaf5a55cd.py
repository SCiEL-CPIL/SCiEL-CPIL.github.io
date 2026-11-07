"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead optimizer for BBOB-style Katsuura function."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def counted_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= self.budget:
                raise RuntimeError("Budget exceeded")
            if time.time() > deadline - 2:
                raise RuntimeError("Time limit")
            calls_made += 1
            val = float(func(x))
            if val < best_f:
                best_f = val
                best_x = np.array(x, copy=True)
            return val

        dim = self.dim
        budget = self.budget

        def nelder_mead(x0, max_calls):
            nonlocal calls_made, best_f, best_x
            n = dim
            # Build initial simplex
            simplex = [np.array(x0, dtype=float, copy=True)]
            for i in range(n):
                xi = np.array(x0, dtype=float, copy=True)
                step = 0.1 if abs(xi[i]) < 0.1 else 0.05 * abs(xi[i])
                xi[i] += step
                simplex.append(xi)
            simplex = [np.clip(x, -5, 5) for x in simplex]

            # Evaluate initial simplex
            fvals = []
            for x in simplex:
                try:
                    fvals.append(counted_func(x))
                except RuntimeError:
                    return

            alpha, gamma, rho = 1.0, 2.0, -0.5

            while calls_made < budget and time.time() < deadline - 5:
                # Sort by fitness
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Check convergence
                if max(fvals) - min(fvals) < 1e-10:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                try:
                    fr = counted_func(xr)
                except RuntimeError:
                    return

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    try:
                        fe = counted_func(xe)
                    except RuntimeError:
                        return
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
                        xc = centroid + 0.5 * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - 0.5 * (simplex[-1] - centroid)
                    xc = np.clip(xc, -5, 5)
                    try:
                        fc = counted_func(xc)
                    except RuntimeError:
                        return
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + 0.5 * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                        try:
                            fvals[i] = counted_func(simplex[i])
                        except RuntimeError:
                            return

        # Multi-start strategy
        # First, do a quick random search to find a good starting point
        n_initial = 30
        for i in range(n_initial):
            if calls_made >= budget or time.time() > deadline - 5:
                break
            x = rng.uniform(-5, 5, dim)
            try:
                counted_func(x)
            except RuntimeError:
                break

        # Then run Nelder-Mead from the best point and several others
        n_restarts = 10
        remaining_budget = budget - calls_made
        calls_per_restart = max(10, remaining_budget // n_restarts)

        # Start from best point found so far
        if best_x is not None:
            nelder_mead(best_x, calls_per_restart)

        # Start from other good points
        for i in range(n_restarts - 1):
            if calls_made >= budget or time.time() > deadline - 5:
                break
            x = rng.uniform(-5, 5, dim)
            nelder_mead(x, calls_per_restart)

        # Final polish from the best point
        if best_x is not None and calls_made < budget:
            nelder_mead(best_x, budget - calls_made)

        return best_x, best_f
# EVOLVE-BLOCK-END
