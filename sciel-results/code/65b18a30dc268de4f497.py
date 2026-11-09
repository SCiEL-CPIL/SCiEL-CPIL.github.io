"""Optimizer for BBOB f13 (Sharp Ridge) with Nelder-Mead + restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead with adaptive restarts and coarse-to-fine strategy."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = time.time() + 280  # safety margin before 300s limit

    def _call(self, func, x):
        """Evaluate function with call counting and deadline check."""
        if self.calls >= self.budget:
            return None
        if time.time() > self.deadline:
            return None
        self.calls += 1
        fx = func(x)
        if fx < self.best_f:
            self.best_f = fx
            self.best_x = x.copy()
        return fx

    def _nelder_mead(self, func, x0, budget):
        """Nelder-Mead derivative-free optimization with adaptive simplex."""
        dim = self.dim
        # Initialize simplex
        simplex = [x0.copy()]
        fvals = [self._call(func, x0)]
        if fvals[0] is None:
            return

        # Expand simplex
        for i in range(dim):
            xi = x0.copy()
            if i == 0:
                xi[0] += 0.5
            else:
                xi[i] += 0.5
            simplex.append(np.clip(xi, -5, 5))
            fvals.append(self._call(func, simplex[-1]))
            if fvals[-1] is None:
                return

        alpha = 1.0    # reflection
        gamma = 2.0    # expansion
        rho = 0.5      # contraction
        sigma = 0.5    # shrink

        max_iter = budget - self.calls
        for _ in range(max_iter):
            if self.calls >= self.budget or time.time() > self.deadline:
                break

            # Sort by function value
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = self._call(func, xr)
            if fr is None:
                break

            if fvals[0] <= fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            elif fr < fvals[0]:
                # Try expansion
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, -5, 5)
                fe = self._call(func, xe)
                if fe is None:
                    break
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            else:
                # Contraction
                if fr < fvals[-1]:
                    # Outside contraction
                    xc = centroid + rho * (xr - centroid)
                else:
                    # Inside contraction
                    xc = centroid - rho * (xr - centroid)
                xc = np.clip(xc, -5, 5)
                fc = self._call(func, xc)
                if fc is None:
                    break

                if fc <= min(fr, fvals[-1]):
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fi = self._call(func, simplex[i])
                        if fi is None:
                            return
                        fvals[i] = fi

        return

    def __call__(self, func):
        dim = self.dim
        rng = np.random.default_rng(42)

        # Phase 1: Coarse exploration with multiple random starts
        n_coarse = 10
        coarse_budget = max(30, self.budget // 5)

        for i in range(n_coarse):
            if self.calls >= self.budget or time.time() > self.deadline:
                break
            x0 = rng.uniform(-5, 5, dim)
            self._nelder_mead(func, x0, coarse_budget)

        # Phase 2: Fine refinement around best point
        if self.best_x is not None and self.calls < self.budget:
            remaining = self.budget - self.calls
            # Multiple restarts near best
            n_fine = min(5, remaining // 20)
            for i in range(n_fine):
                if self.calls >= self.budget or time.time() > self.deadline:
                    break
                # Perturb best point slightly
                x0 = self.best_x + rng.normal(0, 0.1, dim)
                x0 = np.clip(x0, -5, 5)
                self._nelder_mead(func, x0, remaining - self.calls)

        # Phase 3: Final local refinement
        if self.best_x is not None and self.calls < self.budget:
            self._nelder_mead(func, self.best_x, self.budget - self.calls)

        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
