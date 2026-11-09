"""Bent-cigar (f12) black-box optimizer: adaptive coordinate descent + line search + multi-start."""
import time
import numpy as np


class Optimizer:
    """Anisotropy-aware coordinate descent with adaptive step sizes,
    adaptive coordinate ordering, and a few restarts. Robust for f12."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 280.0          # safety margin well under 300 s
        calls = 0

        def f(x):
            nonlocal calls
            calls += 1
            return float(func(x))

        best_x = np.zeros(dim)
        best_f = np.inf

        # ---- 1-D line search (golden section) along axis `ax` from base step `step` ----
        def line_search(x, fx, ax, step, max_iter=18):
            nonlocal calls
            lo, hi = -step, step
            s = step
            while s <= 8.0 and calls < budget and time.time() < deadline:
                xl = x.copy(); xl[ax] += lo; xl = np.clip(xl, -5, 5)
                xr = x.copy(); xr[ax] += hi; xr = np.clip(xr, -5, 5)
                fl = f(xl); fr = f(xr)
                if fl < fx and fr < fx:
                    break
                lo, hi, s = lo * 2.0, hi * 2.0, s * 2.0
            g = (np.sqrt(5.0) - 1.0) / 2.0
            a, b = lo, hi
            c = b - g * (b - a); d = a + g * (b - a)
            xc = x.copy(); xc[ax] = c; xc = np.clip(xc, -5, 5)
            xd = x.copy(); xd[ax] = d; xd = np.clip(xd, -5, 5)
            fc = f(xc); fd = f(xd)
            for _ in range(max_iter):
                if calls >= budget or time.time() >= deadline:
                    break
                if fc < fd:
                    b, d, fd = d, c, fc
                    c = b - g * (b - a)
                    xc = x.copy(); xc[ax] = c; xc = np.clip(xc, -5, 5)
                    fc = f(xc)
                else:
                    a, c, fc = c, d, fd
                    d = a + g * (b - a)
                    xd = x.copy(); xd[ax] = d; xd = np.clip(xd, -5, 5)
                    fd = f(xd)
            cand = x.copy(); cf = fx
            if fc < cf:
                cand[ax] = c; cf = fc
            if fd < cf:
                cand[ax] = d; cf = fd
            return cand, cf

        # ---- one coordinate-descent sweep over `order`, adaptive steps ----
        def cd_sweep(x, fx, steps, order):
            nonlocal calls, best_x, best_f
            improved_any = False
            for ax in order:
                if calls >= budget or time.time() >= deadline:
                    return x, fx, order, improved_any
                nx, nfx = line_search(x, fx, ax, steps[ax])
                if nfx < fx - 1e-12:
                    x, fx = nx, nfx
                    steps[ax] = min(steps[ax] * 1.3, 4.0)
                    improved_any = True
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    steps[ax] = max(steps[ax] * 0.7, 1e-4)
            order = sorted(order, key=lambda a: -steps[a])
            return x, fx, order, improved_any

        # ---- multi-start: origin first, then a few spread seeds ----
        rng = np.random.default_rng(0)
        seeds = [np.zeros(dim)]
        for _ in range(4):
            seeds.append(rng.uniform(-3.5, 3.5, dim))

        steps = np.ones(dim) * 1.0
        for x0 in seeds:
            if calls >= budget or time.time() >= deadline:
                break
            x = np.clip(x0, -5, 5).copy()
            fx = f(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            improved_this_start = fx < best_f + 1e-12
            order = list(range(dim))
            # coordinate descent until stagnation or budget
            for _ in range(300):
                if calls >= budget or time.time() >= deadline:
                    break
                x, fx, order, improved = cd_sweep(x, fx, steps, order)
                if not improved:
                    # small nudge to escape a flat valley
                    if calls + dim <= budget and time.time() < deadline:
                        moved = False
                        for ax in range(dim):
                            nx = x.copy()
                            nx[ax] += steps[ax] * 0.5 * (1.0 if (ax % 2 == 0) else -1.0)
                            nx = np.clip(nx, -5, 5)
                            nfx = f(nx)
                            if nfx < fx:
                                x, fx = nx, nfx
                                if fx < best_f:
                                    best_f = fx
                                    best_x = x.copy()
                                moved = True
                                break
                        if not moved:
                            break
                    else:
                        break
        return best_x, best_f
