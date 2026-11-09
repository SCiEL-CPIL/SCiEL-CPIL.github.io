"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Nelder-Mead simplex method with adaptive restarts for BBOB f17."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initial simplex
        x0 = rng.uniform(-5, 5, dim)
        fx0 = func(x0)

        # Build simplex around x0
        simplex = np.empty((dim + 1, dim))
        fvals = np.empty(dim + 1)
        simplex[0] = x0
        fvals[0] = fx0

        scale = 0.5
        for i in range(1, dim + 1):
            pt = x0.copy()
            if i == 1:
                pt[0] += scale
            else:
                pt[i - 1] += scale
            simplex[i] = np.clip(pt, -5, 5)
            fvals[i] = func(simplex[i])

        calls_made = dim + 1

        # Nelder-Mead parameters
        alpha = 1.0    # reflection
        gamma = 2.0    # expansion
        rho = 0.5      # contraction
        sigma = 0.5    # shrink

        # Track best
        best_x = simplex[np.argmin(fvals)].copy()
        best_f = fvals.min()

        # Adaptive restart: if progress stalls, restart with smaller sigma or new center
        stall_count = 0
        stall_threshold = 20
        min_improvement = 1e-8

        for _ in range(budget - calls_made):
            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

            # Update best
            if fvals[0] < best_f - min_improvement:
                best_f = fvals[0]
                best_x = simplex[0].copy()
                stall_count = 0
            else:
                stall_count += 1

            # Check for convergence
            if stall_count > 100:
                # Restart: perturb the best point
                break

            centroid = np.mean(simplex[1:], axis=0)

            # Reflection
            xr = np.clip(centroid + alpha * (centroid - simplex[0]), -5, 5)
            fr = func(xr)
            calls_made += 1

            if fr < fvals[0]:
                # Expansion
                xe = np.clip(centroid + gamma * (xr - centroid), -5, 5)
                fe = func(xe)
                calls_made += 1
                if fe < fr:
                    simplex[0] = xe
                    fvals[0] = fe
                else:
                    simplex[0] = xr
                    fvals[0] = fr
            elif fr < fvals[-1]:
                # Just reflection
                simplex[0] = xr
                fvals[0] = fr
            else:
                # Contraction
                if fr < fvals[0]:
                    # Outside contraction
                    xc = np.clip(centroid + rho * (xr - centroid), -5, 5)
                else:
                    # Inside contraction
                    xc = np.clip(centroid - rho * (centroid - simplex[0]), -5, 5)
                fc = func(xc)
                calls_made += 1

                if fc <= min(fr, fvals[0] if fr >= fvals[0] else fr):
                    simplex[0] = xc
                    fvals[0] = fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), -5, 5)
                        fvals[i] = func(simplex[i])
                        calls_made += 1

            if calls_made >= budget:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
