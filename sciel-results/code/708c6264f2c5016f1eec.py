"""Combined optimizer for BBOB f12 (Bent cigar) using Nelder-Mead simplex with multi-start."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0  # safety margin before 330s limit

        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls = 0

        def safe_eval(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return float('inf')
            x = np.clip(x, -5, 5)
            f = float(func(x))
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        def nelder_mead(start_x, max_iter=500):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return

            # Initialize simplex
            x0 = np.clip(start_x, -5, 5).copy()
            f0 = safe_eval(x0)
            if calls >= budget or time.time() > deadline:
                return

            # Build initial simplex
            simplex = [x0.copy()]
            simplex_f = [f0]
            for i in range(dim):
                xi = x0.copy()
                # Adaptive initial step size
                if abs(x0[i]) > 1e-10:
                    xi[i] += 0.05 * abs(x0[i])
                else:
                    xi[i] += 0.05
                xi = np.clip(xi, -5, 5)
                fi = safe_eval(xi)
                if calls >= budget or time.time() > deadline:
                    return
                simplex.append(xi)
                simplex_f.append(fi)

            # Nelder-Mead parameters
            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma = 0.5   # shrink

            for _ in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(simplex_f)
                simplex = [simplex[i] for i in order]
                simplex_f = [simplex_f[i] for i in order]

                # Check convergence
                if len(simplex) < 2:
                    break
                f_range = simplex_f[-1] - simplex_f[0]
                if f_range < 1e-12 * max(1.0, abs(simplex_f[0])):
                    break

                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5, 5)
                f_ref = safe_eval(x_ref)
                if calls >= budget or time.time() > deadline:
                    break

                if f_ref < simplex_f[0]:
                    # Try expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5, 5)
                    f_exp = safe_eval(x_exp)
                    if calls >= budget or time.time() > deadline:
                        break
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                elif f_ref < simplex_f[-1]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < simplex_f[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5, 5)
                    f_con = safe_eval(x_con)
                    if calls >= budget or time.time() > deadline:
                        break

                    if f_con <= min(f_ref, simplex_f[-1]):
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = safe_eval(simplex[i])
                            if calls >= budget or time.time() > deadline:
                                return

        # --- Phase 1: Multi-start initialization ---
        # Evaluate origin first
        f_origin = safe_eval(np.zeros(dim))
        if calls >= budget:
            return best_x, best_f

        # Try several random starting points and run Nelder-Mead from the best ones
        n_starts = 8
        start_points = [np.zeros(dim)]
        for _ in range(n_starts - 1):
            if calls >= budget or time.time() > deadline:
                break
            x_rand = rng.uniform(-4, 4, dim)
            start_points.append(x_rand)

        # Evaluate all start points
        start_f = []
        for sp in start_points:
            if calls >= budget or time.time() > deadline:
                break
            f_sp = safe_eval(sp)
            start_f.append(f_sp)

        # Sort start points by function value
        if start_f:
            order = np.argsort(start_f)
            sorted_starts = [start_points[i] for i in order]
            sorted_f = [start_f[i] for i in order]

            # Run Nelder-Mead from the best few starting points
            n_nm_runs = min(4, len(sorted_starts))
            for i in range(n_nm_runs):
                if calls >= budget or time.time() > deadline:
                    break
                nelder_mead(sorted_starts[i], max_iter=300)

        # If budget remains, do a few more random restarts
        if calls < budget - 50 and time.time() < deadline:
            for _ in range(5):
                if calls >= budget - 20 or time.time() > deadline:
                    break
                x_rand = rng.uniform(-5, 5, dim)
                f_rand = safe_eval(x_rand)
                if f_rand < best_f:
                    nelder_mead(x_rand, max_iter=200)

        return best_x, best_f
