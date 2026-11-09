"""Two-phase: coarse Sobol scan with center-bias + multi-start Nelder-Mead refinement."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Coarse Sobol scan followed by multi-start Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _sobol(self, n, dim, seed):
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
            return sampler.random(n)
        except Exception:
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

    def _nelder_mead(self, func, x0, calls_left, deadline, best_state, initial_step=1.0):
        dim = self.dim
        n = dim + 1
        simplex = np.empty((n, dim))
        simplex[0] = x0.copy()
        for i in range(1, n):
            simplex[i] = x0.copy()
            delta = initial_step if x0[i-1] < 4.5 else -initial_step
            simplex[i, i-1] = np.clip(x0[i-1] + delta, -5, 5)

        fvals = np.empty(n)
        for i in range(n):
            fvals[i] = func(simplex[i])
            calls_left[0] -= 1
            if fvals[i] < best_state[0]:
                best_state[0] = fvals[i]
                best_state[1] = simplex[i].copy()

        alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
        max_iter = max(1, min(150, calls_left[0] // (dim + 2)))

        for _ in range(max_iter):
            if time.time() >= deadline or calls_left[0] <= 0:
                break
            idx = np.argsort(fvals)
            simplex = simplex[idx]
            fvals = fvals[idx]
            centroid = np.mean(simplex[:-1], axis=0)

            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = func(xr)
            calls_left[0] -= 1
            if fr < best_state[0]:
                best_state[0] = fr
                best_state[1] = xr.copy()

            if fr < fvals[0]:
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, -5, 5)
                fe = func(xe)
                calls_left[0] -= 1
                if fe < best_state[0]:
                    best_state[0] = fe
                    best_state[1] = xe.copy()
                simplex[-1] = xe if fe < fr else xr
                fvals[-1] = fe if fe < fr else fr
            elif fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            else:
                if fr < fvals[-1]:
                    xc = centroid + rho * (xr - centroid)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    calls_left[0] -= 1
                    if fc < best_state[0]:
                        best_state[0] = fc
                        best_state[1] = xc.copy()
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    for i in range(1, n):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        fvals[i] = func(simplex[i])
                        calls_left[0] -= 1
                        if fvals[i] < best_state[0]:
                            best_state[0] = fvals[i]
                            best_state[1] = simplex[i].copy()
                    if calls_left[0] <= 0:
                        break
        return simplex[0].copy(), fvals[0]

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0
        dim = self.dim
        budget = self.budget
        calls_left = [budget]

        def guarded(x):
            if calls_left[0] <= 0:
                return float('inf')
            calls_left[0] -= 1
            return float(func(x))

        # Phase 1: Coarse Sobol scan with center-biased ordering
        n_scan = min(150, budget // 3)
        if n_scan < 20:
            n_scan = 20
        u = self._sobol(n_scan, dim, 42)
        pts = -5.0 + 10.0 * u
        pts = np.clip(pts, -5, 5)

        # Evaluate in center-biased order to find good points early
        dists = np.sum(pts ** 2, axis=1)
        order = np.argsort(dists)

        vals = np.full(n_scan, np.inf)
        best_val = float('inf')
        best_x = None
        for i in order:
            if time.time() >= deadline or calls_left[0] <= 0:
                break
            vals[i] = guarded(pts[i])
            if vals[i] < best_val:
                best_val = vals[i]
                best_x = pts[i].copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_val = guarded(best_x)

        best_state = [best_val, best_x.copy()]

        # Phase 2: Multi-start Nelder-Mead on top candidates
        # Sort by actual function values (only for evaluated points)
        evaluated_mask = vals < np.inf
        if np.any(evaluated_mask):
            eval_indices = np.where(evaluated_mask)[0]
            eval_vals = vals[evaluated_mask]
            top_k = min(5, len(eval_indices))
            top_order = eval_indices[np.argsort(eval_vals)[:top_k]]
        else:
            top_order = [order[0]]

        # Refine top candidates with decreasing initial steps
        steps = [1.0, 0.5, 0.25, 0.1, 0.05]
        for rank, idx in enumerate(top_order):
            if calls_left[0] < 2 * dim + 2 or time.time() >= deadline:
                break
            step = steps[rank] if rank < len(steps) else 0.05
            self._nelder_mead(guarded, pts[idx], calls_left, deadline, best_state, initial_step=step)

        return best_state[1], best_state[0]
# EVOLVE-BLOCK-END
