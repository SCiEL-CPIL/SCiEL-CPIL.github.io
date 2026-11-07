"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive population-based optimizer for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        d = self.dim
        budget = self.budget

        def time_ok():
            return time.time() < deadline

        rng = np.random.default_rng(42)

        # Phase 1: Initial diverse sampling
        n_init = 20
        points = [rng.uniform(-5, 5, d) for _ in range(n_init)]
        
        best_x = points[0].copy()
        best_f = func(best_x)
        calls = 1

        fits = [best_f]
        for x in points[1:]:
            if not time_ok() or calls >= budget:
                break
            fx = func(x)
            calls += 1
            fits.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort by fitness
        idx = np.argsort(fits)
        sorted_points = [points[i] for i in idx]
        sorted_fits = [fits[i] for i in idx]

        # Phase 2: Nelder-Mead style local refinement from best starts
        # Use a population of simplex vertices for each refinement
        def nelder_mead(func, x0, budget_left, time_check, rng, d):
            """Simple Nelder-Mead variant with adaptive step."""
            if budget_left <= 0 or not time_check():
                return x0, func(x0), 0

            # Build initial simplex
            simplex = [x0.copy()]
            fvals = [func(x0)]
            calls_used = 1

            # Initial step size based on box size
            step = 0.5
            for i in range(d):
                xi = x0.copy()
                xi[i] += step
                xi = np.clip(xi, -5, 5)
                if budget_left - calls_used <= 0 or not time_check():
                    break
                fi = func(xi)
                calls_used += 1
                simplex.append(xi)
                fvals.append(fi)

            # Nelder-Mead iterations
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink

            max_iter = min(budget_left - calls_used, 200)
            for _ in range(max_iter):
                if calls_used >= budget_left or not time_check():
                    break

                # Sort simplex
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflect
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = func(x_ref)
                calls_used += 1

                if f_ref < fvals[0]:
                    # Try expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    if calls_used < budget_left and time_check():
                        f_exp = func(x_exp)
                        calls_used += 1
                        if f_exp < f_ref:
                            simplex[-1] = x_exp
                            fvals[-1] = f_exp
                        else:
                            simplex[-1] = x_ref
                            fvals[-1] = f_ref
                    else:
                        simplex[-1] = x_ref
                        fvals[-1] = f_ref
                elif f_ref < fvals[-2]:
                    simplex[-1] = x_ref
                    fvals[-1] = f_ref
                else:
                    # Contraction
                    x_con = centroid + rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = func(x_con)
                    calls_used += 1

                    if f_con < fvals[-1]:
                        simplex[-1] = x_con
                        fvals[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if calls_used < budget_left and time_check():
                                fvals[i] = func(simplex[i])
                                calls_used += 1

                # Check for convergence
                if max(fvals) - min(fvals) < 1e-10:
                    break

            best_idx = np.argmin(fvals)
            return simplex[best_idx], fvals[best_idx], calls_used

        # Refine from top 5 start points
        remaining = budget - calls
        per_start = max(1, remaining // 5)

        for i in range(min(5, len(sorted_points))):
            if not time_ok() or calls >= budget:
                break
            if sorted_fits[i] > best_f + 50:
                continue

            x0 = sorted_points[i]
            f0 = sorted_fits[i]
            x_new, f_new, used = nelder_mead(func, x0, remaining, time_ok, rng, d)
            calls += used
            remaining -= used

            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()

        # Phase 3: Fine-tuning with small perturbations around best
        if not time_ok() or calls >= budget:
            return best_x, best_f

        # Adaptive fine-tuning
        step_sizes = [0.1, 0.05, 0.02, 0.01, 0.005, 0.001]
        for step in step_sizes:
            if not time_ok() or calls >= budget:
                break
            for _ in range(5):
                if not time_ok() or calls >= budget:
                    break
                for dim_idx in range(d):
                    if not time_ok() or calls >= budget:
                        break
                    for sign in [1, -1]:
                        if not time_ok() or calls >= budget:
                            break
                        y = best_x.copy()
                        y[dim_idx] += sign * step
                        y = np.clip(y, -5, 5)
                        fy = func(y)
                        calls += 1
                        if fy < best_f:
                            best_f = fy
                            best_x = y.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
