"""Two-phase optimizer: sparse coordinate-descent basin-finding + Nelder-Mead local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Phase 1: sparse coordinate descent (adaptive step) + few restarts.
       Phase 2: dense Nelder-Mead simplex local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        DEADLINE = t0 + 280.0  # safety margin before 300s wall-clock kill

        calls = [0]

        def f(x):
            calls[0] += 1
            return float(func(x))

        def in_budget():
            return calls[0] < budget

        def time_ok():
            return time.time() < DEADLINE

        def both_ok():
            return in_budget() and time_ok()

        # ---- Phase 1: sparse coordinate descent with adaptive step + restarts ----
        def coord_descent(x0, n_sweeps):
            x = np.clip(x0.copy(), -5.0, 5.0)
            fx = f(x)
            step = 2.0
            for _ in range(n_sweeps):
                if not both_ok():
                    break
                improved_any = False
                for d in range(dim):
                    if not both_ok():
                        break
                    best = fx
                    best_x = x
                    # try positive and negative steps along this coordinate
                    for sgn in (1.0, -1.0):
                        xx = x.copy()
                        xx[d] = np.clip(x[d] + sgn * step, -5.0, 5.0)
                        fy = f(xx)
                        if fy < best:
                            best = fy
                            best_x = xx
                    if best < fx:
                        x = best_x
                        fx = best
                        improved_any = True
                if improved_any:
                    step = min(step * 1.3, 8.0)
                else:
                    step *= 0.5
                    if step < 0.05:
                        break
            return x, fx

        # a few random restarts to find a promising basin
        best_x = None
        best_fx = np.inf
        n_restart = 4
        per_restart_budget = max(1, (budget // 2) // n_restart)  # reserve half for phase 2
        for r in range(n_restart):
            if not both_ok():
                break
            x0 = np.random.default_rng(1000 + r).uniform(-5.0, 5.0, dim)
            x, fx = coord_descent(x0, per_restart_budget)
            if fx < best_fx:
                best_fx = fx
                best_x = x

        # ---- Phase 2: Nelder-Mead simplex local refinement ----
        def nelder_mead(x0, fx0, max_iter=400):
            x = np.clip(x0.copy(), -5.0, 5.0)
            fx = fx0
            # build simplex: center + unit-ish perturbations
            simplex = [x.copy()]
            for d in range(dim):
                pt = x.copy()
                pt[d] = np.clip(pt[d] + 0.5, -5.0, 5.0)
                simplex.append(pt)
            fvals = [fx] + [f(p) for p in simplex[1:]]
            alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
            for _ in range(max_iter):
                if not both_ok():
                    break
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                centroid = np.mean(simplex[:-1], axis=0)
                worst = simplex[-1]
                # reflect
                xr = centroid + alpha * (centroid - worst)
                xr = np.clip(xr, -5.0, 5.0)
                fr = f(xr)
                if fr < fvals[0]:
                    # expand
                    xe = centroid + gamma * (centroid - worst)
                    xe = np.clip(xe, -5.0, 5.0)
                    fe = f(xe)
                    if fe < fr:
                        simplex[-1], fvals[-1] = xe, fe
                    else:
                        simplex[-1], fvals[-1] = xr, fr
                elif fr < fvals[-2]:
                    simplex[-1], fvals[-1] = xr, fr
                else:
                    # contract
                    if fr < fvals[-1]:
                        xc = centroid + rho * (worst - centroid)
                    else:
                        xc = centroid - rho * (worst - centroid)
                    xc = np.clip(xc, -5.0, 5.0)
                    fc = f(xc)
                    if fc < fvals[-1]:
                        simplex[-1], fvals[-1] = xc, fc
                    else:
                        # shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            fvals[i] = f(simplex[i])
                # track best
                bi = int(np.argmin(fvals))
                if fvals[bi] < fx:
                    fx = fvals[bi]
                    x = simplex[bi].copy()
            return x, fx

        if best_x is not None and both_ok():
            x2, fx2 = nelder_mead(best_x, best_fx, max_iter=300)
            if fx2 < best_fx:
                best_x, best_fx = x2, fx2

        return best_x, best_fx
# EVOLVE-BLOCK-END
