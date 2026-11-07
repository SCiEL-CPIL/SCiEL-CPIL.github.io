"""Adaptive coordinate-descent with multi-restart and Nelder-Mead polish for Schwefel f20."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-restart adaptive coordinate-descent with Nelder-Mead polish on [-5,5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 300.0  # safety margin before 330s kill
        calls = 0

        def F(x):
            nonlocal calls
            calls += 1
            return func(x)

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf

        def time_ok():
            return time.time() < deadline - 5.0

        def evaluate(x):
            nonlocal best_x, best_f
            fx = F(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def adaptive_coord_descent(x0, f0, budget_alloc):
            """Adaptive coordinate descent with per-dimension step sizes."""
            nonlocal best_x, best_f, calls
            n = dim
            x = x0.copy()
            fx = f0
            if fx < best_f:
                best_f, best_x = fx, x.copy()

            # Initialize step sizes
            steps = np.full(n, 1.0)
            min_step = 1e-6
            max_step = 5.0

            no_improve_count = 0
            max_no_improve = 15

            while calls < budget_alloc and time_ok():
                improved_this_round = False

                for d in range(n):
                    if calls >= budget_alloc or not time_ok():
                        break

                    for sign in [1, -1]:
                        if calls >= budget_alloc or not time_ok():
                            break
                        y = x.copy()
                        y[d] += sign * steps[d]
                        y = np.clip(y, -5, 5)
                        fy = F(y)

                        if fy < fx:
                            x = y.copy()
                            fx = fy
                            steps[d] = min(steps[d] * 1.3, max_step)
                            improved_this_round = True
                            no_improve_count = 0
                            if fx < best_f:
                                best_f, best_x = fx, x.copy()
                        else:
                            steps[d] = max(steps[d] * 0.7, min_step)

                if not improved_this_round:
                    no_improve_count += 1
                    if no_improve_count >= max_no_improve:
                        # Random perturbation to escape local minimum
                        x = x + rng.standard_normal(n) * 0.5
                        x = np.clip(x, -5, 5)
                        fx = F(x)
                        if fx < best_f:
                            best_f, best_x = fx, x.copy()
                        no_improve_count = 0
                        steps = np.maximum(steps, 0.5)

                    # If steps are too small, break
                    if np.all(steps < min_step * 10):
                        break
                else:
                    no_improve_count = 0

            return x, fx

        def nelder_mead_polish(x0, f0, budget_alloc):
            """Nelder-Mead simplex method for local refinement."""
            nonlocal best_x, best_f, calls
            n = dim
            if calls >= budget_alloc or not time_ok():
                return x0, f0

            # Initialize simplex
            simplex = [x0.copy()]
            fvals = [f0]
            if f0 < best_f:
                best_f, best_x = f0, x0.copy()

            # Add n more vertices
            for i in range(n):
                xi = x0.copy()
                if i == 0:
                    xi[i] += 0.1
                else:
                    xi[i] += 0.1
                    xi[i - 1] -= 0.1
                xi = np.clip(xi, -5, 5)
                fi = F(xi)
                simplex.append(xi)
                fvals.append(fi)
                if fi < best_f:
                    best_f, best_x = fi, xi.copy()

                if calls >= budget_alloc or not time_ok():
                    break

            # Nelder-Mead parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            max_iter = min(budget_alloc - calls, 200)

            for _ in range(max_iter):
                if calls >= budget_alloc or not time_ok():
                    break

                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst point
                centroid = np.mean(simplex[:n], axis=0)

                # Reflection
                xr = centroid + alpha * (centroid - simplex[n])
                xr = np.clip(xr, -5, 5)
                fr = F(xr)
                if fr < best_f:
                    best_f, best_x = fr, xr.copy()

                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[n])
                    xe = np.clip(xe, -5, 5)
                    fe = F(xe)
                    if fe < best_f:
                        best_f, best_x = fe, xe.copy()
                    if fe < fr:
                        simplex[n] = xe
                        fvals[n] = fe
                    else:
                        simplex[n] = xr
                        fvals[n] = fr
                elif fr < fvals[n - 1]:
                    simplex[n] = xr
                    fvals[n] = fr
                else:
                    # Contraction
                    if fr < fvals[n]:
                        xc = centroid + rho * (simplex[n] - centroid)
                    else:
                        xc = centroid - rho * (centroid - simplex[n])
                    xc = np.clip(xc, -5, 5)
                    fc = F(xc)
                    if fc < best_f:
                        best_f, best_x = fc, xc.copy()
                    if fc < max(fr, fvals[n]):
                        simplex[n] = xc
                        fvals[n] = fc
                    else:
                        # Shrink
                        for i in range(1, n + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            fvals[i] = F(simplex[i])
                            if fvals[i] < best_f:
                                best_f, best_x = fvals[i], simplex[i].copy()

            return best_x, best_f

        # Phase 1: Multi-restart adaptive coordinate descent
        # Use ~80% of budget for multiple CD restarts
        cd_budget_total = int(budget * 0.8)
        n_restarts = 6
        budget_per_restart = cd_budget_total // n_restarts

        # Generate diverse starting points
        start_points = []
        # Center
        start_points.append(np.zeros(dim))
        # Random points in [-4, 4]^dim
        for _ in range(n_restarts - 1):
            start_points.append(rng.uniform(-4, 4, dim))

        for i, sp in enumerate(start_points):
            if calls >= cd_budget_total or not time_ok():
                break
            remaining_for_this = min(budget_per_restart, cd_budget_total - calls)
            if remaining_for_this < 10:
                break
            # Evaluate starting point
            f_sp = F(sp)
            if f_sp < best_f:
                best_f, best_x = f_sp, sp.copy()
            # Run adaptive CD from this point
            adaptive_coord_descent(sp, f_sp, remaining_for_this)

        # Phase 2: Nelder-Mead polish from best point
        remaining = budget - calls
        if remaining > 20 and time_ok():
            nelder_mead_polish(best_x.copy(), best_f, remaining)

        return best_x, best_f
# EVOLVE-BLOCK-END
