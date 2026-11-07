"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with adaptive restarts and deadline guard for sharp ridge f13."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return best_f
            x = np.clip(x, -5, 5)
            try:
                fx = float(func(x))
            except Exception:
                return np.inf
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Nelder-Mead implementation
        def nelder_mead(start, max_iter=200):
            nonlocal calls_made, best_x, best_f
            dim = self.dim
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            # Initialize simplex
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = start
            for i in range(1, dim + 1):
                simplex[i] = start.copy()
                simplex[i, i - 1] = 0.25  # small perturbation along axis i-1

            # Evaluate initial simplex
            for i in range(dim + 1):
                simplex[i] = np.clip(simplex[i], -5, 5)
                fvals = np.array([safe_call(simplex[i]) for i in range(dim + 1)])

            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

            best_local = fvals[0]
            best_local_x = simplex[0].copy()

            for _ in range(max_iter):
                if time.time() > deadline - 2.0:
                    break
                if calls_made >= budget:
                    break

                # Update global best
                if best_local < best_f:
                    best_f = best_local
                    best_x = best_local_x.copy()

                # Check convergence
                if np.max(fvals) - np.min(fvals) < 1e-10:
                    break

                # Centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = safe_call(x_ref)

                if fvals[0] <= f_ref < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                elif f_ref < fvals[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = safe_call(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        fvals[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < fvals[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = safe_call(x_con)

                    if f_con <= min(f_ref, fvals[-1]):
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = safe_call(simplex[i])

                # Re-sort
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                best_local = fvals[0]
                best_local_x = simplex[0].copy()

            return best_local_x, best_local

        # Multi-start strategy: try a few different starting points
        n_starts = 3
        for s in range(n_starts):
            if time.time() > deadline - 5.0:
                break
            if calls_made >= budget:
                break

            if s == 0:
                start = rng.uniform(-3, 3, dim)
            else:
                start = rng.uniform(-5, 5, dim)

            x_opt, f_opt_local = nelder_mead(start, max_iter=150)

            if f_opt_local < best_f:
                best_f = f_opt_local
                best_x = x_opt.copy()

            # If we've made good progress, refine around the best point
            if best_f < 1.0 and calls_made < budget - 50:
                # Do a finer local search around the best point
                fine_start = best_x + rng.normal(0, 0.1, dim)
                x_fine, f_fine = nelder_mead(fine_start, max_iter=100)
                if f_fine < best_f:
                    best_f = f_fine
                    best_x = x_fine.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
