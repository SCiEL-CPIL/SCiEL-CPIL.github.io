"""Rewritten optimizer for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: Halton space-filling exploration + adaptive Nelder-Mead refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _halton(self, index, base):
        """Compute the Halton sequence value for a given index and base."""
        result = 0.0
        f = 1.0
        i = index
        while i > 0:
            f /= base
            result += f * (i % base)
            i //= base
        return result

    def _halton_sequence(self, n, dim, offset=0):
        """Generate n points in [0,1)^dim using Halton sequence."""
        points = np.zeros((n, dim))
        for i in range(n):
            for j in range(dim):
                points[i, j] = self._halton(i + 1 + offset, j + 2)
        return points

    def _nelder_mead(self, func, x0, max_iter=500, tol=1e-10, deadline=None):
        """
        Adaptive Nelder-Mead simplex method.
        Returns (best_x, best_f, calls_used).
        """
        dim = len(x0)
        # Initialize simplex
        simplex = np.zeros((dim + 1, dim))
        simplex[0] = x0
        for i in range(dim):
            simplex[i + 1] = x0.copy()
            simplex[i + 1][i] += 0.1

        fvals = np.array([func(simplex[i]) for i in range(dim + 1)])
        calls = dim + 1

        # Adaptive step size
        step = 0.5
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrink

        best_x = simplex[np.argmin(fvals)]
        best_f = fvals.min()

        for iteration in range(max_iter):
            if deadline is not None and time.time() > deadline:
                break

            # Sort simplex by function value
            order = np.argsort(fvals)
            simplex = simplex[order]
            fvals = fvals[order]

            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()

            # Check convergence
            if np.max(fvals) - np.min(fvals) < tol:
                break

            # Centroid of all but worst point
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = func(xr)
            calls += 1

            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                fe = func(xe)
                calls += 1
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + 0.5 * (simplex[-1] - centroid)
                else:
                    # Inside contraction
                    xc = centroid - 0.5 * (simplex[-1] - centroid)
                fc = func(xc)
                calls += 1
                if fc <= min(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
                        calls += 1
                    if deadline is not None and time.time() > deadline:
                        break

            # Adaptive step adjustment
            if calls > 0:
                progress = best_f - fvals[0]
                if progress < 1e-8:
                    step *= 0.95
                else:
                    step = min(step * 1.05, 2.0)

            if deadline is not None and time.time() > deadline:
                break

        return best_x, best_f, calls

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Phase 1: Halton space-filling search for broad coverage
        # Use first ~100 evaluations for systematic exploration
        n_halton = min(80, budget // 5)
        halton_points = self._halton_sequence(n_halton, dim)
        # Map from [0,1)^dim to [-5,5]^dim
        halton_points = halton_points * 10.0 - 5.0

        best_x = None
        best_f = np.inf
        calls = 0

        # Evaluate Halton points
        for i in range(n_halton):
            if calls >= budget or time.time() > deadline:
                break
            x = halton_points[i]
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: Nelder-Mead refinement from best Halton point
        if best_x is not None and calls < budget:
            nm_deadline = deadline - 5  # Leave 5s buffer
            x_nm, f_nm, nm_calls = self._nelder_mead(
                func, best_x, max_iter=200, tol=1e-12, deadline=nm_deadline
            )
            calls += nm_calls
            if f_nm < best_f:
                best_f = f_nm
                best_x = x_nm

        # Phase 3: If budget remains, try a few more Halton points with different offsets
        remaining = budget - calls
        if remaining > 10 and time.time() < deadline:
            # Try a few more strategic points around the best found
            n_extra = min(remaining // 2, 20)
            if n_extra > 0:
                extra_points = self._halton_sequence(n_extra, dim, offset=n_halton)
                extra_points = extra_points * 10.0 - 5.0
                for i in range(n_extra):
                    if calls >= budget or time.time() > deadline:
                        break
                    x = extra_points[i]
                    fx = func(x)
                    calls += 1
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                        # Quick NM refinement if we found something better
                        if calls < budget - 20 and time.time() < deadline - 10:
                            x_nm2, f_nm2, nm_calls2 = self._nelder_mead(
                                func, best_x, max_iter=50, tol=1e-10, deadline=deadline - 10
                            )
                            calls += nm_calls2
                            if f_nm2 < best_f:
                                best_f = f_nm2
                                best_x = x_nm2

        return best_x, best_f
# EVOLVE-BLOCK-END
