"""Optimizer for BBOB f9 (rotated Rosenbrock). Fresh design:
global multi-start (Halton + golden-section radius) + dense-Hessian Newton
polish. Time-guarded so it always returns the best point found."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import approx_fprime, approx_hess


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0          # safety margin vs 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls = 0

        def f(x):
            nonlocal calls
            calls += 1
            return float(func(np.asarray(x, dtype=float)))

        def in_time():
            return time.time() < deadline

        # ---------- Newton polish with dense Hessian (Rosenbrock-friendly) ----
        def polish(x, fval):
            nonlocal best_x, best_f, calls
            x = np.asarray(x, float).copy()
            fv = fval
            for _ in range(200):
                if not in_time():
                    break
                if calls >= budget - 2:
                    break
                g = approx_fprime(x, f, 1e-5)
                if np.max(np.abs(g)) < 1e-12:
                    break
                H = approx_hess(x, f, 1e-5)
                H = H + 1e-8 * np.eye(dim)
                try:
                    d = np.linalg.solve(H, -g)
                except np.linalg.LinAlgError:
                    d = -g
                # backtracking line search (box-clipped)
                alpha = 1.0
                improved = False
                for _ in range(30):
                    xn = np.clip(x + alpha * d, -5.0, 5.0)
                    fn = f(xn)
                    if not np.isfinite(fn):
                        alpha *= 0.5
                        continue
                    if fn < fv:
                        x, fv = xn, fn
                        improved = True
                        break
                    alpha *= 0.5
                if not improved:
                    break
            if fv < best_f:
                best_f, best_x = fv, x.copy()
            return x, fv

        # ---------- Global multi-start (Halton + golden-section radius) --------
        def halton(n, base):
            out = np.zeros(n)
            f = 1.0
            for i in range(n):
                x = 0.0
                ii = i + 1
                while ii > 0:
                    f /= base
                    x += f * (ii % base)
                    ii //= base
                out[i] = x
            return out

        # golden-section search over radius to maximize the best found
        def radius_eval(r):
            n = max(1, int(8 * r + 1))
            if n > 120:
                n = 120
            pts = np.tile(halton(n, 3), (dim, 1)).T * (2 * r) - r
            pts = np.clip(pts, -5.0, 5.0)
            pts[0] = np.zeros(dim)
            best_r = np.inf
            for p in pts:
                if not in_time():
                    break
                if calls >= budget - 50:
                    break
                v = f(p)
                if v < best_r:
                    best_r = v
            return best_r

        lo, hi = 0.0, 5.0
        gr = (np.sqrt(5.0) - 1.0) / 2.0
        c1 = hi - gr * (hi - lo)
        c2 = lo + gr * (hi - lo)
        f1, f2 = radius_eval(c1), radius_eval(c2)
        for _ in range(8):
            if not in_time() or calls >= budget - 60:
                break
            if f1 < f2:
                hi, f2, c2, f2 = c2, f2, c1, f1
                c1 = hi - gr * (hi - lo)
                f1 = radius_eval(c1)
            else:
                lo, f1, c1, f1 = c1, f1, c2, f2
                c2 = lo + gr * (hi - lo)
                f2 = radius_eval(c2)

        # ---------- Polish the best points from the global sweep --------------
        n = 40
        pts = np.tile(halton(n, 3), (dim, 1)).T * (2 * 5.0) - 5.0
        pts[0] = np.zeros(dim)
        # add the best radius's points again
        rbest = (lo + hi) / 2.0
        pts2 = np.tile(halton(max(1, int(8 * rbest + 1)), 3), (dim, 1)).T * (2 * rbest) - rbest
        pts = np.vstack([pts, np.clip(pts2, -5.0, 5.0)])

        # sort by value (already have values? recompute cheaply for ordering)
        vals = np.array([f(p) for p in pts] if in_time() and calls < budget - 5 else np.full(len(pts), np.inf))
        order = np.argsort(vals)
        for idx in order:
            if not in_time() or calls >= budget - 10:
                break
            p = np.clip(pts[idx], -5.0, 5.0)
            if not np.isfinite(vals[idx]):
                continue
            polish(p, vals[idx])

        # ---------- Final safeguard: polish global best ------------------------
        if best_x is not None and in_time() and calls < budget:
            polish(best_x, best_f)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = f(best_x)
        return best_x, best_f
# EVOLVE-BLOCK-END
