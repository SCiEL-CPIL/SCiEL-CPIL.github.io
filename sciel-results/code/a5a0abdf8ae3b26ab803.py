"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with adaptive restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 270  # safety margin before 300s limit

        call_count = [0]
        best_x = None
        best_f = np.inf

        def counted_func(x):
            nonlocal best_x, best_f
            if call_count[0] >= budget:
                return best_f
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            call_count[0] += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        def nelder_mead(x0, max_iter=50):
            """Nelder-Mead simplex method."""
            nonlocal best_x, best_f
            x0 = np.clip(x0, -5.0, 5.0).copy()
            f0 = counted_func(x0)
            if best_x is None:
                best_x = x0.copy()
                best_f = f0

            # Initialize simplex
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            for i in range(dim):
                step = np.zeros(dim)
                step[i] = 0.5
                simplex[i + 1] = x0 + step

            fvals = np.array([counted_func(simplex[i]) for i in range(dim + 1)])

            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            for _ in range(max_iter):
                if call_count[0] >= budget:
                    break
                if time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]

                # Update global best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = simplex[0].copy()

                # Check convergence
                if np.max(fvals) - np.min(fvals) < 1e-10:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5.0, 5.0)
                fr = counted_func(xr)

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = counted_func(xe)
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
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = counted_func(xc)
                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fvals[i] = counted_func(simplex[i])

            return best_x, best_f

        # Multi-start strategy
        # Use diverse initial points to explore the landscape
        n_starts = 5
        starts_per = max(1, (budget - 10) // (n_starts * 50))

        for s in range(n_starts):
            if call_count[0] >= budget or time.time() > deadline:
                break

            if s == 0:
                x0 = np.zeros(dim)
            elif s == 1:
                x0 = rng.uniform(-3, 3, dim)
            elif s == 2:
                x0 = rng.uniform(-4, 4, dim)
            elif s == 3:
                # Perturb current best
                if best_x is not None:
                    x0 = np.clip(best_x + 0.5 * rng.standard_normal(dim), -5, 5)
                else:
                    x0 = rng.uniform(-4, 4, dim)
            else:
                x0 = rng.uniform(-5, 5, dim)

            nelder_mead(x0, max_iter=starts_per)

        # Final polish: local refinement around best found
        if best_x is not None and call_count[0] < budget and time.time() < deadline:
            # Use a smaller step Nelder-Mead
            for _ in range(3):
                if call_count[0] >= budget or time.time() > deadline:
                    break
                nelder_mead(best_x, max_iter=10)

        return best_x, best_f
# EVOLVE-BLOCK-END
