# EVOLVE-BLOCK-START
import numpy as np
import time


def _cd_polish(func, x, budget, deadline, iters=60, step=0.6):
    """Deterministic coordinate-descent line search. Cheap, gives early progress."""
    x = np.array(x, dtype=float).copy()
    fx = func(x)
    n = len(x)
    for _ in range(iters):
        if time.time() >= deadline:
            break
        improved = False
        for d in range(n):
            if time.time() >= deadline:
                break
            s = step
            for _ in range(40):
                if time.time() >= deadline:
                    break
                y = x.copy()
                y[d] = x[d] + s
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    improved = True
                    s *= 0.5
                else:
                    y[d] = x[d] - s
                    fy = func(y)
                    if fy < fx:
                        x, fx = y, fy
                        improved = True
                        s *= 0.5
                    else:
                        break
        if not improved:
            break
    return x, fx


def _nelder_mead(func, x0, budget, deadline, iters=400, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.7):
    """Nelder-Mead simplex; adapts geometry to the sharp ridge."""
    n = len(x0)
    x0 = np.array(x0, dtype=float)
    simplex = np.empty((n + 1, n))
    simplex[0] = x0
    for i in range(1, n + 1):
        simplex[i] = x0.copy()
        simplex[i, i - 1] += 0.5
    fvals = np.array([func(simplex[i]) for i in range(n + 1)])
    best_x = simplex.copy()
    best_f = fvals.copy()
    for _ in range(iters):
        if time.time() >= deadline:
            break
        order = np.argsort(fvals)
        simplex = simplex[order]
        fvals = fvals[order]
        best_x = simplex.copy()
        best_f = fvals.copy()
        if np.max(fvals) - np.min(fvals) < 1e-10 * (1.0 + abs(np.min(fvals))):
            break
        centroid = simplex[:-1].mean(axis=0)
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = func(xr)
        if fr < fvals[0]:
            xe = centroid + gamma * (xr - centroid)
            fe = func(xe)
            if fe < fr:
                simplex[-1], fvals[-1] = xe, fe
            else:
                simplex[-1], fvals[-1] = xr, fr
        elif fr < fvals[-2]:
            simplex[-1], fvals[-1] = xr, fr
        else:
            xc = centroid + sigma * (simplex[-1] - centroid)
            fc = func(xc)
            if fc < fvals[-1]:
                simplex[-1], fvals[-1] = xc, fc
            else:
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + rho * (simplex[i] - simplex[0])
                    fvals[i] = func(simplex[i])
    return best_x, np.min(best_f)


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        start = time.time()
        deadline = start + 280.0  # safety margin before 330s wall-clock kill

        # Phase 1: cheap coordinate descent from origin for fast early progress
        x, fx = _cd_polish(func, np.zeros(dim), budget=30, deadline=deadline, iters=30, step=2.0)

        # Phase 2: Nelder-Mead from the CD result (adapts to ridge geometry)
        x2, fx2 = _nelder_mead(func, x, budget=120, deadline=deadline, iters=120)
        if fx2 < fx:
            x, fx = x2, fx2

        # Phase 3: diversify — Nelder-Mead from a few deterministic spread points
        # to escape the sharp ridge's narrow valley if we plateaued
        candidates = [np.zeros(dim)]
        for c in range(8):
            v = np.array([(-1.0) ** (c >> i) & 1 for i in range(dim)])
            v = v / np.linalg.norm(v)
            candidates.append(v * 3.5)
        for c in candidates[1:]:
            if time.time() >= deadline:
                break
            xc, fxc = _nelder_mead(func, c, budget=60, deadline=deadline, iters=60)
            if fxc < fx:
                x, fx = xc, fxc

        # Phase 4: final coordinate-descent polish to squeeze out the last bits
        x, fx = _cd_polish(func, x, budget=40, deadline=deadline, iters=40, step=0.4)

        return x, fx
# EVOLVE-BLOCK-END
