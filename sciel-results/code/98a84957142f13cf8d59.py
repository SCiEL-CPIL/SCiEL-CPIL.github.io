"""Nelder-Mead based optimizer with restarts for BBOB f13 (Sharp Ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex optimizer with adaptive restarts and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit, 20s safety margin
        n = self.dim
        calls = 0
        best_x = None
        best_f = np.inf

        def safe_func(x):
            nonlocal calls, best_x, best_f
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def nelder_mead(func_eval, x0, initial_alpha=0.05, max_iter=None):
            """Run Nelder-Mead from x0, returning best point found."""
            nonlocal calls, deadline
            n = self.dim
            # Build initial simplex
            simplex = [x0.copy()]
            for i in range(n):
                pt = x0.copy()
                pt[i] += initial_alpha
                pt = np.clip(pt, -5.0, 5.0)
                simplex.append(pt)

            fvals = [func_eval(simplex[i]) for i in range(n + 1)]

            # Nelder-Mead parameters
            alpha_r = 1.0   # reflection
            gamma_e = 2.0   # expansion
            rho_c = 0.5     # contraction
            sigma_s = 0.5   # shrink

            it = 0
            max_iter = max_iter or (self.budget - calls - 5)
            no_improve = 0
            prev_best = np.inf

            for _ in range(max_iter):
                if calls >= self.budget or time.time() > deadline:
                    break

                it += 1
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                best_val = fvals[0]
                worst_val = fvals[-1]

                # Check convergence
                if abs(worst_val - best_val) < 1e-10:
                    break

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha_r * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = func_eval(x_r)

                if f_r < fvals[0]:
                    # Try expansion
                    x_e = centroid + gamma_e * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = func_eval(x_e)
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                    no_improve = 0
                elif f_r < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                    no_improve = 0
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        # Outside contraction
                        x_c = centroid + rho_c * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - rho_c * (simplex[-1] - centroid)
                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = func_eval(x_c)

                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                        no_improve = 0
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma_s * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fvals[i] = func_eval(simplex[i])
                        no_improve += 1

                if no_improve > 15:
                    break

            return simplex[np.argmin(fvals)], min(fvals)

        # Strategy: multiple restarts with different initial points
        # Use a mix of random and structured initializations
        rng = np.random.default_rng(123)

        # First try: Nelder-Mead from center
        x0 = np.zeros(n)
        x_best, f_best = nelder_mead(safe_func, x0, initial_alpha=0.5)
        if f_best < best_f:
            best_f = f_best
            best_x = x_best.copy()

        # If we have budget left, try more restarts
        if calls < self.budget and time.time() < deadline:
            # Try from the best found so far with smaller alpha
            if best_x is not None:
                x_best2, f_best2 = nelder_mead(safe_func, best_x, initial_alpha=0.1)
                if f_best2 < best_f:
                    best_f = f_best2
                    best_x = x_best2.copy()

        # More restarts from random points if budget allows
        restarts = 0
        while calls < self.budget - 10 and time.time() < deadline and restarts < 8:
            # Generate a new starting point
            # Mix: some near best, some random
            if restarts % 2 == 0 and best_x is not None:
                x0 = best_x + rng.normal(0, 0.5, n)
            else:
                x0 = rng.uniform(-4, 4, n)
            x0 = np.clip(x0, -5.0, 5.0)

            x_new, f_new = nelder_mead(safe_func, x0, initial_alpha=0.3)
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
            restarts += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
