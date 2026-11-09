"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-phase: Nelder-Mead simplex + adaptive coordinate descent refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _nelder_mead(self, func, x0, budget, deadline, rng):
        """Nelder-Mead simplex method with adaptive step sizes."""
        n = self.dim
        # Initialize simplex
        simplex = [x0.copy()]
        fvals = [func(x0)]
        for i in range(n):
            dx = x0.copy()
            dx[i] += 0.5
            simplex.append(np.clip(dx, -5, 5))
            fvals.append(func(simplex[-1]))

        alpha, gamma, rho = 1.0, 2.0, 0.5
        best_idx = 0
        for i in range(1, n + 1):
            if fvals[i] < fvals[best_idx]:
                best_idx = i

        calls = 1 + n
        for _ in range(budget - calls):
            if time.time() > deadline:
                break

            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            best_idx = 0

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
            fr = func(xr)
            calls += 1

            if fr < fvals[0]:
                # Expansion
                xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5, 5)
                fe = func(xe)
                calls += 1
                if fe < fr:
                    simplex[-1], fvals[-1] = xe, fe
                else:
                    simplex[-1], fvals[-1] = xr, fr
            elif fr < fvals[-2]:
                simplex[-1], fvals[-1] = xr, fr
            else:
                # Contraction
                xc = np.clip(centroid + rho * (simplex[-1] - centroid), -5, 5)
                fc = func(xc)
                calls += 1
                if fc < fvals[-1]:
                    simplex[-1], fvals[-1] = xc, fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = np.clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]), -5, 5)
                        fvals[i] = func(simplex[i])
                        calls += 1
                        if time.time() > deadline:
                            break
                    if time.time() > deadline:
                        break

        # Return best point
        order = np.argsort(fvals)
        return simplex[order[0]], fvals[order[0]], calls

    def _coordinate_descent(self, func, x, budget, deadline, rng):
        """Adaptive coordinate descent with backtracking."""
        n = self.dim
        best_x = x.copy()
        best_f = func(x)
        calls = 1
        step_sizes = np.ones(n) * 1.0

        for _ in range(budget - calls):
            if time.time() > deadline:
                break

            improved = False
            for i in range(n):
                if time.time() > deadline:
                    break

                # Try positive and negative steps
                for direction in [1, -1]:
                    dx = np.zeros(n)
                    dx[i] = direction * step_sizes[i]
                    x_try = np.clip(best_x + dx, -5, 5)
                    f_try = func(x_try)
                    calls += 1

                    if f_try < best_f:
                        best_x = x_try
                        best_f = f_try
                        step_sizes[i] *= 1.3
                        improved = True
                        break
                else:
                    # No improvement in either direction, shrink step
                    step_sizes[i] *= 0.7

            if not improved:
                # Global shrink
                step_sizes *= 0.8
                if np.all(step_sizes < 1e-6):
                    break

        return best_x, best_f, calls

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        # Phase 1: Nelder-Mead with ~70% of budget
        nm_budget = int(self.budget * 0.7)
        x0 = np.array([0.0] * self.dim)  # Start at origin
        x, fx, calls_used = self._nelder_mead(func, x0, nm_budget, deadline, rng)

        # Phase 2: Coordinate descent refinement with remaining budget
        cd_budget = self.budget - calls_used
        if cd_budget > 10 and time.time() < deadline:
            x, fx, calls_cd = self._coordinate_descent(func, x, cd_budget, deadline, rng)

        return x, fx
# EVOLVE-BLOCK-END
