"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Grid-guided multi-restart Nelder-Mead for Rosenbrock-style function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, max_calls, deadline):
        """Run Nelder-Mead simplex method with a strict call budget."""
        n = self.dim
        # Initial simplex
        simplex = [x0.copy()]
        for i in range(n):
            v = x0.copy()
            step = 0.3
            v[i] += step
            simplex.append(np.clip(v, -5, 5))

        # Evaluate initial simplex
        fvals = []
        for v in simplex:
            if max_calls <= 0:
                fvals.append(np.inf)
                continue
            max_calls -= 1
            fvals.append(func(v))

        if any(np.isinf(f) for f in fvals):
            return x0, fvals[0], 0

        best_idx = np.argmin(fvals)
        best_f = fvals[best_idx]
        best_x = simplex[best_idx].copy()

        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5

        calls_used = n + 1
        while max_calls > 0 and time.time() < deadline:
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            centroid = np.mean(simplex[:-1], axis=0)

            xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
            fr = func(xr)
            max_calls -= 1
            calls_used += 1

            if fr < fvals[0]:
                xe = np.clip(centroid + gamma * (xr - centroid), -5, 5)
                fe = func(xe)
                max_calls -= 1
                calls_used += 1
                if fe < fr:
                    simplex[-1], fvals[-1] = xe, fe
                    if fe < best_f:
                        best_f, best_x = fe, xe.copy()
                else:
                    simplex[-1], fvals[-1] = xr, fr
                    if fr < best_f:
                        best_f, best_x = fr, xr.copy()
            elif fr < fvals[-2]:
                simplex[-1], fvals[-1] = xr, fr
                if fr < best_f:
                    best_f, best_x = fr, xr.copy()
            else:
                if fr < fvals[-1]:
                    xc = np.clip(centroid + rho * (xr - centroid), -5, 5)
                else:
                    xc = np.clip(centroid - rho * (simplex[-1] - centroid), -5, 5)
                fc = func(xc)
                max_calls -= 1
                calls_used += 1

                if fc < fvals[-1]:
                    simplex[-1], fvals[-1] = xc, fc
                    if fc < best_f:
                        best_f, best_x = fc, xc.copy()
                else:
                    for i in range(1, n + 1):
                        if max_calls <= 0:
                            break
                        simplex[i] = np.clip(simplex[0] + sigma * (simplex[i] - simplex[0]), -5, 5)
                        fvals[i] = func(simplex[i])
                        max_calls -= 1
                        calls_used += 1
                        if fvals[i] < best_f:
                            best_f, best_x = fvals[i], simplex[i].copy()

            spread = max(fvals) - min(fvals)
            if spread < 1e-12:
                break

        return best_x, best_f, calls_used

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made
            if calls_made >= self.budget or time.time() >= deadline:
                return None
            calls_made += 1
            try:
                return func(np.clip(x, -5, 5))
            except Exception:
                return None

        # Phase 1: Coarse structured grid search over [-4, 4]^5
        # Use 3 levels per dimension: -4, 0, 4
        grid_levels = np.array([-4.0, 0.0, 4.0])
        grid_points = []
        for x0 in grid_levels:
            for x1 in grid_levels:
                for x2 in grid_levels:
                    for x3 in grid_levels:
                        for x4 in grid_levels:
                            grid_points.append(np.array([x0, x1, x2, x3, x4]))

        # Add some intermediate points for better coverage
        grid_points.append(np.zeros(self.dim))
        grid_points.append(np.ones(self.dim) * 2.0)
        grid_points.append(np.ones(self.dim) * -2.0)
        grid_points.append(np.array([2.0, 2.0, 2.0, 2.0, 2.0]))
        grid_points.append(np.array([-2.0, -2.0, -2.0, -2.0, -2.0]))
        grid_points.append(np.array([3.0, 3.0, 3.0, 3.0, 3.0]))
        grid_points.append(np.array([-3.0, -3.0, -3.0, -3.0, -3.0]))

        # Evaluate grid points (limited to ~100 calls for grid phase)
        grid_budget = min(len(grid_points), 100)
        grid_results = []
        for i in range(grid_budget):
            x = grid_points[i]
            f = safe_call(x)
            if f is not None:
                grid_results.append((f, x.copy()))
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

        if not grid_results:
            return best_x, best_f

        # Sort by function value
        grid_results.sort(key=lambda x: x[0])

        # Phase 2: Nelder-Mead refinement from top grid points
        remaining = self.budget - calls_made
        if remaining <= 0:
            return best_x, best_f

        # Allocate budget to top 3-5 grid points
        n_starts = min(len(grid_results), max(3, len(grid_results) // 4))
        budget_per_start = remaining // n_starts

        for k in range(n_starts):
            if calls_made >= self.budget or time.time() >= deadline:
                break
            if budget_per_start < 15:
                break

            _, x0 = grid_results[k]
            budget_for_this = min(budget_per_start, self.budget - calls_made)

            bx, bf, used = self._nelder_mead(func, x0, budget_for_this, deadline)
            calls_made += used

            if bf < best_f:
                best_f = bf
                best_x = bx.copy()

        # Phase 3: Final refinement from global best
        if best_x is not None and calls_made < self.budget and time.time() < deadline:
            remaining = self.budget - calls_made
            if remaining > 10:
                bx, bf, used = self._nelder_mead(func, best_x, remaining, deadline)
                calls_made += used
                if bf < best_f:
                    best_f = bf
                    best_x = bx.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
