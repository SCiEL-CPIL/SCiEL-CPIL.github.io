"""Fresh design: Sobol global scan + Nelder-Mead local refinement, time-guarded."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Deterministic Sobol scan followed by Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    # ---- helpers -------------------------------------------------------
    def _sobol(self, n, dim, rng):
        # Use scipy if available for a true Sobol sequence; else fall back to
        # a well-spaced Halton-like construction.
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=dim, scramble=True, seed=rng)
            pts = sampler.random(n)  # in [0,1)^dim
            return pts
        except Exception:
            # Halton fallback
            def halton(index, base):
                f, r = 1.0, 0.0
                while index > 0:
                    f /= base
                    r += f * (index % base)
                    index //= base
                return r
            pts = np.empty((n, dim))
            for d in range(dim):
                for i in range(n):
                    pts[i, d] = halton(i + 1, d + 2)
            return pts

    def _nelder_mead(self, func, x0, calls_left, deadline, best_state):
        # Simplex method with adaptive step, budget + time guarded.
        dim = self.dim
        n = dim + 1
        simplex = np.empty((n, dim))
        simplex[0] = x0.copy()
        for i in range(1, n):
            simplex[i] = x0.copy()
            simplex[i, i - 1] = x0[i - 1] + 0.5 if x0[i - 1] < 4.5 else x0[i - 1] - 0.5
            simplex[i, i - 1] = np.clip(simplex[i, i - 1], -5, 5)

        fvals = np.empty(n)
        for i in range(n):
            fvals[i] = func(simplex[i])
            calls_left[0] -= 1
            if fvals[i] < best_state[0]:
                best_state[0] = fvals[i]
                best_state[1] = simplex[i].copy()

        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        max_iter = max(1, min(200, calls_left[0] // (dim + 1)))

        for _ in range(max_iter):
            if time.time() >= deadline or calls_left[0] <= 0:
                break
            idx = np.argsort(fvals)
            simplex = simplex[idx]
            fvals = fvals[idx]
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls_left[0] -= 1
            if fr < best_state[0]:
                best_state[0] = fr
                best_state[1] = xr.copy()

            if fr < fvals[0]:
                # Expansion
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls_left[0] -= 1
                if fe < best_state[0]:
                    best_state[0] = fe
                    best_state[1] = xe.copy()
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
                    xc = centroid + rho * (xr - centroid)
                    fc = func(xc)
                    calls_left[0] -= 1
                    if fc < best_state[0]:
                        best_state[0] = fc
                        best_state[1] = xc.copy()
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
                        calls_left[0] -= 1
                        if fvals[i] < best_state[0]:
                            best_state[0] = fvals[i]
                            best_state[1] = simplex[i].copy()
                    if calls_left[0] <= 0:
                        break
        return simplex[0].copy(), fvals[0]

    # ---- main ----------------------------------------------------------
    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 300s kill
        dim = self.dim
        budget = self.budget
        calls_left = [budget]

        def guarded(x):
            if calls_left[0] <= 0:
                return float('inf')
            calls_left[0] -= 1
            return float(func(x))

        rng = np.random.default_rng(12345)

        # Phase 1: Sobol global scan (use a fixed fraction of budget)
        n_scan = min(120, budget // 4)
        if n_scan < 10:
            n_scan = 10
        u = self._sobol(n_scan, dim, rng)
        pts = -5.0 + 10.0 * u  # map to [-5,5]^dim
        pts = np.clip(pts, -5, 5)

        best_val = float('inf')
        best_x = None
        order = np.argsort(np.sum((pts - 0.0) ** 2, axis=1))  # center-biased
        for i in order:
            if time.time() >= deadline or calls_left[0] <= 0:
                break
            fx = guarded(pts[i])
            if fx < best_val:
                best_val = fx
                best_x = pts[i].copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_val = guarded(best_x)

        # Phase 2: local refinement on top candidates
        # Sort scan points by value; we need values, so re-evaluate is wasteful.
        # Instead keep track: we only stored best. Do a small re-scan of a subset
        # to get top-k. To avoid extra calls, we'll just polish the global best
        # and a few deterministic diverse starts.
        best_state = [best_val, best_x.copy()]

        # Polish the best point
        if calls_left[0] > 2 * dim + 2 and time.time() < deadline:
            self._nelder_mead(guarded, best_x, calls_left, deadline, best_state)

        # A couple of diverse local starts from other Sobol points near the best
        # (to hedge against a slightly-off global pick), only if budget remains.
        if calls_left[0] > 2 * dim + 2 and time.time() < deadline:
            # pick 2 more points that are reasonably good: re-evaluate a small set
            cand_idx = list(order[:min(5, n_scan)])
            cand_idx = [i for i in cand_idx if not np.allclose(pts[i], best_x)]
            for i in cand_idx[:2]:
                if calls_left[0] <= 2 * dim + 2 or time.time() >= deadline:
                    break
                fx = guarded(pts[i])
                if fx < best_state[0] * 1.5 + 1.0:  # plausibly good basin
                    self._nelder_mead(guarded, pts[i], calls_left, deadline, best_state)

        return best_state[1], best_state[0]
# EVOLVE-BLOCK-END
