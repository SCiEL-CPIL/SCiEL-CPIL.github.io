"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with LHS initialization for BBOB f1 (Sphere)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Call counter
        calls = [0]

        def wrapped(x):
            calls[0] += 1
            return func(np.asarray(x, dtype=np.float64))

        # LHS initialization
        rng = np.random.default_rng(42)
        x0 = self._lhs(dim, 1)[0]
        f0 = wrapped(x0)
        best_x = x0.copy()
        best_f = f0

        # Nelder-Mead with adaptive parameters
        # Initial simplex
        eps = 0.1
        simplex = [x0.copy() + eps * rng.standard_normal(dim) for _ in range(dim)]
        # Ensure first point is x0
        simplex[0] = x0.copy()
        fvals = [f0]
        for i in range(1, dim + 1):
            fvals.append(wrapped(simplex[i]))

        # Track best
        for i in range(dim + 1):
            if fvals[i] < best_f:
                best_f = fvals[i]
                best_x = simplex[i].copy()

        # Nelder-Mead parameters
        alpha = 1.0    # reflection
        gamma = 2.0    # expansion
        rho = 0.5      # contraction
        sigma_nm = 0.5 # shrink
        max_iter = budget - (calls[0])

        for iteration in range(max_iter):
            # Check deadline
            if time.time() > deadline:
                break
            if calls[0] >= budget:
                break

            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i].copy() for i in order]
            fvals = [fvals[i] for i in order]

            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            f_r = wrapped(x_r)

            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - simplex[-1])
                f_e = wrapped(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                if f_r < fvals[-1]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                    f_c = wrapped(x_c)
                    if f_c <= f_r:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            fvals[i] = wrapped(simplex[i])
                else:
                    # Inside contraction
                    x_c = centroid - rho * (centroid - simplex[-1])
                    f_c = wrapped(x_c)
                    if f_c < fvals[-1]:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            fvals[i] = wrapped(simplex[i])

            # Update best after each iteration
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = simplex[min_idx].copy()

            # Check convergence
            if len(set(fvals)) == 1 or max(fvals) - min(fvals) < 1e-15:
                break

        return best_x, best_f

    def _lhs(self, dim, n):
        """Latin Hypercube Sampling."""
        rng = np.random.default_rng(42)
        points = np.zeros((n, dim))
        for d in range(dim):
            perm = rng.permutation(n)
            for i in range(n):
                points[i, d] = (perm[i] + rng.random()) / n
        # Scale to [-5, 5]
        points = points * 10.0 - 5.0
        return points
# EVOLVE-BLOCK-END
