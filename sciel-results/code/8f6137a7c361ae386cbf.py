"""Optimizer for BBOB f19 (Griewank-Rosenbrock) with multi-start Nelder-Mead."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start Nelder-Mead with adaptive simplex and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _nelder_mead(self, func, x0, budget, deadline, rng, call_counter):
        """Run Nelder-Mead simplex optimization from x0."""
        n = self.dim
        # Initialize simplex: x0 and n other points
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0
        for i in range(1, n + 1):
            simplex[i] = x0.copy()
            simplex[i, i - 1] += 0.5

        # Evaluate initial simplex
        fvals = np.zeros(n + 1)
        for i in range(n + 1):
            if call_counter['count'] >= budget or time.time() > deadline:
                break
            x = np.clip(simplex[i], -5.0, 5.0)
            fvals[i] = self._safe_call(func, x, call_counter)

        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink

        best_f = fvals.min()
        best_x = simplex[np.argmin(fvals)].copy()

        max_iter = 2000
        for it in range(max_iter):
            if call_counter['count'] >= budget or time.time() > deadline:
                break

            # Sort vertices by function value
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

            # Check convergence
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()

            if np.std(fvals) < 1e-10 * (1 + abs(fvals[0])):
                break

            # Centroid of all but worst point
            centroid = simplex[:-1].mean(axis=0)

            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, -5.0, 5.0)
            f_r = self._safe_call(func, x_r, call_counter)

            if call_counter['count'] >= budget or time.time() > deadline:
                break

            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (centroid - simplex[-1])
                x_e = np.clip(x_e, -5.0, 5.0)
                f_e = self._safe_call(func, x_e, call_counter)
                if call_counter['count'] >= budget or time.time() > deadline:
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
                    x_c = centroid - rho * (simplex[-1] - centroid)
                x_c = np.clip(x_c, -5.0, 5.0)
                f_c = self._safe_call(func, x_c, call_counter)
                if call_counter['count'] >= budget or time.time() > deadline:
                    break
                if f_c < min(f_r, fvals[-1]):
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        fvals[i] = self._safe_call(func, simplex[i], call_counter)
                        if call_counter['count'] >= budget or time.time() > deadline:
                            break

        return best_x, best_f

    def _safe_call(self, func, x, call_counter):
        """Safely call func and track the best result."""
        x = np.asarray(x, dtype=float).reshape(self.dim)
        x = np.clip(x, -5.0, 5.0)
        fx = float(func(x))
        call_counter['count'] += 1
        if fx < call_counter['best_f']:
            call_counter['best_f'] = fx
            call_counter['best_x'] = x.copy()
        return fx

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        deadline = time.time() + 290.0  # safety margin before 300s kill

        call_counter = {'count': 0, 'best_f': np.inf, 'best_x': None}

        # Stage 1: Coarse random sampling to find good starting points
        n_coarse = min(20, budget)
        coarse_points = []
        for i in range(n_coarse):
            if time.time() > deadline or call_counter['count'] >= budget:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            fx = self._safe_call(func, x, call_counter)
            coarse_points.append((fx, x.copy()))

        if call_counter['count'] >= budget:
            return call_counter['best_x'], call_counter['best_f']

        # Sort coarse points by fitness
        coarse_points.sort(key=lambda t: t[0])

        # Stage 2: Multi-start Nelder-Mead from best coarse points
        n_starts = min(3, len(coarse_points))
        for i in range(n_starts):
            if time.time() > deadline or call_counter['count'] >= budget:
                break
            fx_init, x_init = coarse_points[i]
            best_x, best_f = self._nelder_mead(
                func, x_init, budget, deadline, rng, call_counter
            )
            if best_f < call_counter['best_f']:
                call_counter['best_f'] = best_f
                call_counter['best_x'] = best_x.copy()

        # Stage 3: Final polish with very small steps around best
        if call_counter['best_x'] is not None and call_counter['count'] < budget:
            x = call_counter['best_x'].copy()
            fx = call_counter['best_f']
            for _ in range(min(20, budget - call_counter['count'])):
                if time.time() > deadline or call_counter['count'] >= budget:
                    break
                step = 0.01 * rng.standard_normal(dim)
                y = np.clip(x + step, -5.0, 5.0)
                fy = self._safe_call(func, y, call_counter)
                if fy < fx:
                    x = y.copy()
                    fx = fy

        return call_counter['best_x'], call_counter['best_f']
# EVOLVE-BLOCK-END
