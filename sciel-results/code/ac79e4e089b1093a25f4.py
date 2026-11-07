"""Optimized black-box continuous optimizer for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-phase optimizer: restart-based global search + high-precision local polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.monotonic()
        deadline = start_time + 320.0  # safety margin before the 330s kill
        rng = np.random.default_rng(0)

        # Counted wrapper so we never exceed the budget.
        calls = {"n": 0}

        def cfunc(x):
            if calls["n"] >= budget:
                return np.inf
            calls["n"] += 1
            try:
                v = func(x)
            except Exception:
                return np.inf
            if not np.isfinite(v):
                return np.inf
            return float(v)

        def time_ok():
            return time.monotonic() < deadline and calls["n"] < budget

        # ---- Global phase: restarts with adaptive (1+1)-ES ----
        def es_from(x0, fx, iters, sigma):
            x, f = x0, fx
            s = sigma
            for _ in range(iters):
                if not time_ok():
                    break
                y = np.clip(x + s * rng.standard_normal(dim), -5, 5)
                fy = cfunc(y)
                if fy < f:
                    x, f = y, fy
                    s = min(s * 1.3, 4.0)
                else:
                    s = max(s * 0.85, 1e-3)
            return x, f

        best_x = None
        best_f = np.inf

        # A few exploratory starts with short ES runs.
        n_global = 6
        for k in range(n_global):
            if not time_ok():
                break
            x0 = rng.uniform(-5, 5, dim)
            fx0 = cfunc(x0)
            if fx0 < best_f:
                best_f, best_x = fx0, x0
            x1, f1 = es_from(x0, fx0, iters=40, sigma=2.0)
            if f1 < best_f:
                best_f, best_x = f1, x1

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = cfunc(best_x)

        # ---- Local phase: coordinate descent + Nelder-Mead polish ----
        def coord_descent(x, f, iters, step):
            for _ in range(iters):
                if not time_ok():
                    break
                improved = False
                for d in range(dim):
                    for sign in (1.0, -1.0):
                        y = x.copy()
                        y[d] = np.clip(y[d] + sign * step, -5, 5)
                        fy = cfunc(y)
                        if fy < f:
                            x, f = y, fy
                            improved = True
                if not improved:
                    step *= 0.5
                    if step < 1e-9:
                        break
            return x, f

        def nelder_mead(x, f, max_iter=400):
            # Build simplex around current best.
            simplex = [x.copy()]
            fvals = [f]
            for d in range(dim):
                v = x.copy()
                step = 0.25 if abs(x[d]) < 1e-3 else 0.1 * abs(x[d])
                v[d] = np.clip(v[d] + step, -5, 5)
                fv = cfunc(v)
                simplex.append(v)
                fvals.append(fv)

            alpha, gamma, rho, sigma_ = 1.0, 2.0, -0.5, 0.5
            shrink = 0.5
            for _ in range(max_iter):
                if not time_ok():
                    break
                # Order by fitness.
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                xr = np.clip(centroid + alpha * (centroid - simplex[-1]), -5, 5)
                fr = cfunc(xr)
                if fvals[0] <= fr < fvals[-2]:
                    simplex[-1], fvals[-1] = xr, fr
                    continue
                # Expansion
                if fr < fvals[0]:
                    xe = np.clip(centroid + gamma * (centroid - simplex[-1]), -5, 5)
                    fe = cfunc(xe)
                    if fe < fr:
                        simplex[-1], fvals[-1] = xe, fe
                    else:
                        simplex[-1], fvals[-1] = xr, fr
                    continue
                # Contraction
                if fr < fvals[-1]:
                    xc = np.clip(centroid + rho * (centroid - simplex[-1]), -5, 5)
                    fc = cfunc(xc)
                    if fc < fvals[-1]:
                        simplex[-1], fvals[-1] = xc, fc
                        continue
                # Shrink
                for i in range(1, len(simplex)):
                    simplex[i] = np.clip(simplex[0] + shrink * (simplex[i] - simplex[0]), -5, 5)
                    fvals[i] = cfunc(simplex[i])

                # Convergence check
                spread = max(np.max(np.abs(np.array(simplex) - simplex[0]), axis=0))
                if spread < 1e-9:
                    break

            # Return best vertex.
            bi = int(np.argmin(fvals))
            return simplex[bi], fvals[bi]

        # Polish from the best global point.
        if time_ok():
            px, pf = coord_descent(best_x.copy(), best_f, iters=200, step=0.5)
            if pf < best_f:
                best_f, best_x = pf, px
            if time_ok():
                px, pf = nelder_mead(best_x.copy(), best_f, max_iter=600)
                if pf < best_f:
                    best_f, best_x = pf, px
            if time_ok():
                px, pf = coord_descent(best_x.copy(), best_f, iters=300, step=0.1)
                if pf < best_f:
                    best_f, best_x = pf, px
            if time_ok():
                px, pf = nelder_mead(best_x.copy(), best_f, max_iter=800)
                if pf < best_f:
                    best_f, best_x = pf, px

        return best_x, best_f
# EVOLVE-BLOCK-END
