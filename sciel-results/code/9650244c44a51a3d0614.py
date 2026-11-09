# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-line (1+1)-ES exploration + Nelder-Mead refinement on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)
        # Hard call cap with a small safety margin (must stay <= budget).
        self.max_calls = self.budget - 2
        self._deadline = None

    def __call__(self, func):
        start = time.time()
        self._deadline = start + 290.0  # safety margin vs 300s wall clock
        d = self.dim
        rng = np.random.default_rng(12345)

        self._calls = 0
        self._best_x = None
        self._best_f = np.inf

        def call(x):
            if self._calls >= self.max_calls:
                return None
            self._calls += 1
            return func(np.asarray(x, dtype=float))

        def time_ok():
            return time.time() < self._deadline

        # ---------------- Phase A: diverse multi-line (1+1)-ES ----------------
        n_lines = 8
        # Diverse starts: spread across the box plus a couple of near-origin.
        starts = []
        for i in range(n_lines):
            s = rng.uniform(-5.0, 5.0, d)
            starts.append(s)
        # Ensure at least one start near the origin (minimizers lie in [-4,4]^5).
        starts[0] = rng.uniform(-1.0, 1.0, d)

        # Budget split: give each line a chunk, more to early lines.
        total_line_calls = int(0.6 * self.max_calls)
        chunk = max(8, total_line_calls // n_lines)

        for s in starts:
            if not time_ok() or self._calls >= self.max_calls:
                break
            x = np.clip(s.copy(), -5.0, 5.0)
            fx = call(x)
            if fx is None:
                break
            if fx < self._best_f:
                self._best_f, self._best_x = fx, x.copy()

            sigma = 1.5
            for _ in range(chunk):
                if not time_ok() or self._calls >= self.max_calls:
                    break
                y = np.clip(x + sigma * rng.standard_normal(d), -5.0, 5.0)
                fy = call(y)
                if fy is None:
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.4
                    if fx < self._best_f:
                        self._best_f, self._best_x = fx, x.copy()
                else:
                    sigma *= 0.85
                if sigma < 1e-3:
                    sigma = 1e-3

        # ---------------- Phase B: Nelder-Mead refinement on best point ----------
        if self._best_x is not None and time_ok() and self._calls < self.max_calls:
            self._nelder_mead(func, call, self._best_x, time_ok)

        return self._best_x, self._best_f

    def _nelder_mead(self, func, call, x0, time_ok):
        d = self.dim
        # Build initial simplex around the current best.
        simplex = [x0.copy()]
        fx0 = self._best_f
        base = 0.5
        for i in range(d):
            pt = x0.copy()
            pt[i] = x0[i] + base
            pt = np.clip(pt, -5.0, 5.0)
            simplex.append(pt)

        fvals = [fx0]
        for pt in simplex[1:]:
            fv = call(pt)
            if fv is None:
                return
            fvals.append(fv)

        n_iter = 200
        for _ in range(n_iter):
            if not time_ok() or self._calls >= self.max_calls:
                break
            # Order by function value.
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]
            worst_f = fvals[-1]

            # Reflection.
            alpha = 1.0
            xr = np.clip(centroid + alpha * (centroid - worst), -5.0, 5.0)
            fr = call(xr)
            if fr is None:
                break

            if fr < fvals[0]:
                # Expansion.
                gamma = 2.0
                xe = np.clip(centroid + gamma * (centroid - worst), -5.0, 5.0)
                fe = call(xe)
                if fe is None:
                    break
                if fe < fr:
                    simplex[-1], fvals[-1] = xe, fe
                else:
                    simplex[-1], fvals[-1] = xr, fr
            elif fr < fvals[-2]:
                simplex[-1], fvals[-1] = xr, fr
            else:
                # Contraction.
                rho = 0.5
                xc = np.clip(centroid + rho * (worst - centroid), -5.0, 5.0)
                fc = call(xc)
                if fc is None:
                    break
                if fc < worst_f:
                    simplex[-1], fvals[-1] = xc, fc
                else:
                    # Shrink toward best.
                    for i in range(1, len(simplex)):
                        simplex[i] = np.clip(simplex[0] + 0.5 * (simplex[i] - simplex[0]), -5.0, 5.0)
                        fv = call(simplex[i])
                        if fv is None:
                            return
                        fvals[i] = fv

            # Track global best.
            m = int(np.argmin(fvals))
            if fvals[m] < self._best_f:
                self._best_f, self._best_x = fvals[m], simplex[m].copy()

            # Stagnation: if the simplex is numerically collapsed, stop.
            if np.max(fvals) - np.min(fvals) < 1e-12:
                break
# EVOLVE-BLOCK-END
