"""Optimized Rastrigin f3 solver: dense 1D grid for basin finding + multi-start Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Per-coordinate dense 1D grid search to find the Rastrigin basin, then Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        def try_point(x):
            nonlocal best_x, best_f, calls_made
            if calls_made >= budget or not time_ok():
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Dense per-coordinate 1D grid search to locate the global basin ---
        # Rastrigin: f(x) = A*d + sum(x_i^2 - A*cos(2*pi*x_i)), A=10
        # Period of cosine is 1 in each coordinate. Minima of cos at even integers.
        # With shift s, minimum near s mod 2 (even integer neighborhood).
        # Dense grid with spacing < 0.5 ensures we capture the right basin.

        n_grid = 81  # spacing = 8/80 = 0.1 over [-4, 4]
        grid_vals = np.linspace(-4.0, 4.0, n_grid)

        coord_best = np.zeros(dim)

        for d in range(dim):
            local_best_val = 0.0
            local_best_f = float('inf')
            for gv in grid_vals:
                if not time_ok() or calls_made >= budget:
                    break
                x = np.zeros(dim)
                x[d] = gv
                fx = try_point(x)
                if fx is not None and fx < local_best_f:
                    local_best_f = fx
                    local_best_val = gv
            coord_best[d] = local_best_val

        # Build initial candidate from per-coordinate bests
        x0 = coord_best.copy()
        try_point(x0)

        # --- Phase 2: Coordinate-wise refinement with progressively finer grids ---
        for pass_num in range(5):
            if not time_ok() or calls_made >= budget:
                break
            for d in range(dim):
                if not time_ok() or calls_made >= budget:
                    break
                center = best_x[d]
                if pass_num == 0:
                    half_range = 0.5
                    n_pts = 41
                elif pass_num == 1:
                    half_range = 0.25
                    n_pts = 41
                elif pass_num == 2:
                    half_range = 0.125
                    n_pts = 41
                elif pass_num == 3:
                    half_range = 0.0625
                    n_pts = 41
                else:
                    half_range = 0.03125
                    n_pts = 41

                a = max(-5.0, center - half_range)
                b = min(5.0, center + half_range)
                fine_grid = np.linspace(a, b, n_pts)

                local_best_val = center
                local_best_f = best_f
                for gv in fine_grid:
                    if not time_ok() or calls_made >= budget:
                        break
                    x_try = best_x.copy()
                    x_try[d] = gv
                    fx = try_point(x_try)
                    if fx is not None and fx < local_best_f:
                        local_best_f = fx
                        local_best_val = gv

                best_x[d] = local_best_val

        # --- Phase 3: Nelder-Mead polish with tight convergence ---
        if best_x is not None and time_ok() and calls_made < budget:
            nm_budget = budget - calls_made
            if nm_budget > dim + 2:
                x_nm, f_nm, nm_calls = self._nelder_mead(func, best_x, nm_budget, deadline)
                calls_made += nm_calls
                if f_nm < best_f:
                    best_f = f_nm
                    best_x = x_nm.copy()

        # --- Phase 4: If budget remains, do another Nelder-Mead from a perturbed start ---
        if best_x is not None and time_ok() and calls_made < budget - (dim + 2):
            # Small random perturbation to escape potential local minima
            rng = np.random.RandomState(42)
            x_pert = best_x + rng.normal(0, 0.01, dim)
            x_pert = np.clip(x_pert, -5, 5)
            nm_budget2 = budget - calls_made
            if nm_budget2 > dim + 2:
                x_nm2, f_nm2, nm_calls2 = self._nelder_mead(func, x_pert, nm_budget2, deadline)
                calls_made += nm_calls2
                if f_nm2 < best_f:
                    best_f = f_nm2
                    best_x = x_nm2.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made = 1

        return best_x, best_f

    def _nelder_mead(self, func, x0, budget_left, deadline):
        """Nelder-Mead minimization with tight convergence criteria."""
        dim = len(x0)
        simplex = [x0.copy()]
        f0 = func(np.clip(x0, -5, 5))
        simplex_f = [f0]
        calls = 1

        for i in range(dim):
            if calls >= budget_left or time.time() >= deadline:
                break
            x_i = x0.copy()
            x_i[i] += 0.01
            x_i = np.clip(x_i, -5, 5)
            simplex.append(x_i)
            f_i = func(x_i)
            simplex_f.append(f_i)
            calls += 1

        if len(simplex) < 2:
            return x0, f0, calls

        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        max_iter = max(100, budget_left - (dim + 1))

        for _ in range(max_iter):
            if calls >= budget_left or time.time() >= deadline:
                break

            order = np.argsort(simplex_f)
            simplex = [simplex[i] for i in order]
            simplex_f = [simplex_f[i] for i in order]

            centroid = np.mean(simplex[:-1], axis=0)

            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5, 5)
            f_r = func(x_r)
            calls += 1
            if calls >= budget_left or time.time() >= deadline:
                break

            if f_r < simplex_f[0]:
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, -5, 5)
                f_e = func(x_e)
                calls += 1
                if calls >= budget_left or time.time() >= deadline:
                    break
                if f_e < f_r:
                    simplex[-1] = x_e
                    simplex_f[-1] = f_e
                else:
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
            elif f_r < simplex_f[-2]:
                simplex[-1] = x_r
                simplex_f[-1] = f_r
            else:
                if f_r < simplex_f[-1]:
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    x_c = centroid - rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5, 5)
                f_c = func(x_c)
                calls += 1
                if calls >= budget_left or time.time() >= deadline:
                    break
                if f_c < min(f_r, simplex_f[-1]):
                    simplex[-1] = x_c
                    simplex_f[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        if calls >= budget_left or time.time() >= deadline:
                            break
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex_f[i] = func(np.clip(simplex[i], -5, 5))
                        calls += 1
                    if calls >= budget_left or time.time() >= deadline:
                        break

            # Convergence check: very tight
            spread = max(simplex_f) - min(simplex_f)
            x_spread = np.max(np.abs(np.array(simplex) - simplex[0]))
            if spread < 1e-12 and x_spread < 1e-10:
                break

        min_idx = np.argmin(simplex_f)
        return simplex[min_idx], simplex_f[min_idx], calls
# EVOLVE-BLOCK-END
