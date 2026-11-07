"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex optimizer with deadline guard for BBOB Sphere."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink
        eps = 1e-8

        # Initial simplex: center at origin, step size 1.0
        # This is good since minimizer is in [-4,4]^5
        step = 1.0
        simplex = np.zeros((dim + 1, dim))
        for i in range(dim):
            simplex[i + 1, i] = step
        # simplex[0] is origin, simplex[1..dim] are axis-aligned points

        # Evaluate initial simplex
        fvals = np.zeros(dim + 1)
        calls_made = 0

        for i in range(dim + 1):
            if time.time() > deadline:
                break
            x_i = np.clip(simplex[i], -5, 5)
            fvals[i] = func(x_i)
            calls_made += 1
            if calls_made >= budget:
                break

        if calls_made >= budget:
            best_idx = np.argmin(fvals)
            return simplex[best_idx], fvals[best_idx]

        # Nelder-Mead main loop
        while calls_made < budget:
            if time.time() > deadline:
                break

            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

            # Check convergence
            if np.max(fvals) - np.min(fvals) < 1e-10:
                break

            # Centroid of all points except worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls_made += 1

            if calls_made >= budget:
                break

            if f_r < fvals[0]:
                # Try expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls_made += 1

                if calls_made >= budget:
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                    break

                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                # Accept reflection
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                if f_r < fvals[-1]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (x_r - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls_made += 1

                if calls_made >= budget:
                    if f_c < f_r:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    break

                if f_c < min(f_r, fvals[-1]):
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                    for i in range(1, dim + 1):
                        if time.time() > deadline:
                            break
                        fvals[i] = func(simplex[i])
                        calls_made += 1
                        if calls_made >= budget:
                            break

        # Find best point
        order = np.argsort(fvals)
        best_idx = order[0]
        return simplex[best_idx], fvals[best_idx]
# EVOLVE-BLOCK-END
