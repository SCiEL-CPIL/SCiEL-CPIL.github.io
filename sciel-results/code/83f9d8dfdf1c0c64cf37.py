"""Nelder-Mead + Newton/Hessian optimizer for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid Nelder-Mead + Newton optimizer for BBOB f11 Discus."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._deadline = 0.0

    def __call__(self, func):
        self._deadline = time.time() + 270
        total_budget = self.budget
        n = self.dim
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        calls_used = 0

        def wrapper(x):
            nonlocal calls_used
            if calls_used >= total_budget or time.time() > self._deadline:
                raise StopIteration()
            calls_used += 1
            try:
                val = float(func(x))
                if np.isfinite(val):
                    return val
                return 1e10
            except Exception:
                return 1e10

        # Phase 1: Multi-start Nelder-Mead to find good basins
        starts = [np.zeros(n)]
        for _ in range(5):
            starts.append(rng.uniform(-4, 4, n))

        for x0 in starts:
            if calls_used >= total_budget or time.time() > self._deadline:
                break
            try:
                res = self._nelder_mead(wrapper, x0, n, max_iter=30, xatol=1e-4, fatol=1e-4)
                if res[1] < best_f:
                    best_f = res[1]
                    best_x = res[0].copy()
            except (StopIteration, Exception):
                pass

        # Phase 2: Newton refinement with local Hessian at the best point
        if best_x is not None and calls_used < total_budget and time.time() < self._deadline:
            try:
                best_x, best_f = self._newton_refine(
                    wrapper, best_x, best_f, n, max_iter=20, tol=1e-10
                )
            except (StopIteration, Exception):
                pass

        # Phase 3: Another round of NM from best if budget remains
        if best_x is not None and calls_used < total_budget and time.time() < self._deadline:
            try:
                res = self._nelder_mead(wrapper, best_x, n, max_iter=50, xatol=1e-8, fatol=1e-8)
                if res[1] < best_f:
                    best_f = res[1]
                    best_x = res[0].copy()
            except (StopIteration, Exception):
                pass

        # Phase 4: Final Newton polish
        if best_x is not None and calls_used < total_budget and time.time() < self._deadline:
            try:
                best_x, best_f = self._newton_refine(
                    wrapper, best_x, best_f, n, max_iter=10, tol=1e-12
                )
            except (StopIteration, Exception):
                pass

        if best_x is None:
            best_x = np.zeros(n)
            if calls_used < total_budget:
                try:
                    best_f = float(func(best_x))
                except Exception:
                    best_f = np.inf

        return best_x, best_f

    def _nelder_mead(self, func, x0, n, max_iter=50, xatol=1e-6, fatol=1e-6):
        """Simple Nelder-Mead implementation."""
        alpha = 1.5
        gamma = 2.0
        rho = 0.5
        sigma = 0.5

        simplex = [x0.astype(float)]
        for i in range(n):
            xi = x0.copy()
            step = alpha * 0.05 if abs(x0[i]) < 1e-10 else alpha * abs(x0[i])
            xi[i] += step
            simplex.append(xi)

        fvals = [func(simplex[i]) for i in range(n + 1)]

        for _ in range(max_iter):
            if time.time() > self._deadline:
                raise StopIteration()

            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]

            if abs(fvals[0] - fvals[-1]) < fatol:
                break
            centroid = np.mean(simplex[:-1], axis=0)
            if np.linalg.norm(simplex[0] - simplex[-1]) < xatol:
                break

            xr = centroid + gamma * (centroid - simplex[-1])
            fr = func(xr)

            if fr < fvals[0]:
                xe = centroid + gamma * (xr - centroid)
                fe = func(xe)
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
                xc = centroid + rho * (simplex[-1] - centroid)
                fc = func(xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])

        idx = np.argmin(fvals)
        return simplex[idx], fvals[idx]

    def _newton_refine(self, func, x0, f0, n, max_iter=20, tol=1e-10):
        """Newton's method with finite-difference Hessian."""
        x = x0.copy().astype(float)
        f = f0
        h = 1e-6

        for _ in range(max_iter):
            if time.time() > self._deadline:
                raise StopIteration()

            grad = np.zeros(n)
            hess = np.zeros((n, n))

            for i in range(n):
                xp = x.copy()
                xm = x.copy()
                xp[i] += h
                xm[i] -= h
                f_p = func(xp)
                f_m = func(xm)
                grad[i] = (f_p - f_m) / (2 * h)

                for j in range(i, n):
                    xpp = x.copy()
                    xpm = x.copy()
                    xmp = x.copy()
                    xmm = x.copy()
                    xpp[i] += h
                    xpp[j] += h
                    xpm[i] += h
                    xpm[j] -= h
                    xmp[i] -= h
                    xmp[j] += h
                    xmm[i] -= h
                    xmm[j] -= h
                    f_pp = func(xpp)
                    f_pm = func(xpm)
                    f_mp = func(xmp)
                    f_mm = func(xmm)
                    hess_val = (f_pp - f_pm - f_mp + f_mm) / (4 * h * h)
                    hess[i, j] = hess_val
                    hess[j, i] = hess_val

            if np.linalg.norm(grad) < tol:
                break

            try:
                lam = 1e-6
                hess_reg = hess + lam * np.eye(n)
                dx = np.linalg.solve(hess_reg, -grad)
            except np.linalg.LinAlgError:
                dx = -grad * 0.01

            step_norm = np.linalg.norm(dx)
            if step_norm > 1.0:
                dx = dx / step_norm * 1.0

            x_new = x + dx
            f_new = func(x_new)

            if np.isfinite(f_new) and f_new < f:
                x = x_new
                f = f_new
            else:
                dx = dx * 0.5
                x_new = x + dx
                f_new = func(x_new)
                if np.isfinite(f_new) and f_new < f:
                    x = x_new
                    f = f_new
                else:
                    break

        return x, f
# EVOLVE-BLOCK-END
