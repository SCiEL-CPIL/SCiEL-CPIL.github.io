"""Model-and-optimize black-box optimizer for BBOB f18 (Schaffers F7).

Uses a two-phase strategy:
1. Multi-start Nelder-Mead to rapidly locate the basin of attraction.
2. CMA-ES for precise convergence within the ill-conditioned basin.
"""
import time
import numpy as np


class Optimizer:
    """Two-phase optimizer: multi-start Nelder-Mead followed by CMA-ES."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        WALL = 300.0
        calls = [0]
        n = self.dim

        def time_ok():
            return (time.time() - t0) < WALL

        def safe(x):
            if calls[0] >= self.budget or not time_ok():
                return None
            try:
                v = float(func(np.asarray(x, dtype=float)))
            except Exception:
                v = 1e18
            if not np.isfinite(v):
                v = 1e18
            calls[0] += 1
            return v

        # Phase 1: Multi-start Nelder-Mead
        # Allocate ~40% of budget for initial exploration
        nm_budget = int(self.budget * 0.4)
        nm_calls = 0
        best_x = np.zeros(n)
        best_f = np.inf

        # Initialize with a few diverse starting points
        rng = np.random.default_rng(0)
        starts = [np.zeros(n)]
        for _ in range(2):
            starts.append(rng.uniform(-4, 4, n))

        for start in starts:
            if nm_calls >= nm_budget or not time_ok():
                break
            x, f = self._nelder_mead(func, start, n, budget=nm_budget - nm_calls, safe=safe, time_ok=time_ok, best_f=best_f)
            nm_calls += calls[0] - (self.budget - nm_budget)  # approximate
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 2: CMA-ES from the best point found
        if calls[0] < self.budget and time_ok():
            x, f = self._cma_es(func, best_x, n, budget=self.budget - calls[0], safe=safe, time_ok=time_ok, best_f=best_f)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        return best_x, float(best_f)

    def _nelder_mead(self, func, x0, n, budget, safe, time_ok, best_f):
        """Nelder-Mead simplex method."""
        # Initialize simplex
        simplex = [x0.copy()]
        f_simplex = [best_f]
        for i in range(n):
            xi = x0.copy()
            xi[i] += 0.5 if x0[i] < 0 else -0.5
            xi = np.clip(xi, -5.0, 5.0)
            simplex.append(xi)
            f_simplex.append(best_f)

        # Evaluate initial simplex
        for i in range(n + 1):
            v = safe(simplex[i])
            if v is None:
                return x0, best_f
            f_simplex[i] = v

        # Sort
        order = np.argsort(f_simplex)
        simplex = [simplex[i] for i in order]
        f_simplex = [f_simplex[i] for i in order]

        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        max_iter = budget // (n + 1)

        for _ in range(max_iter):
            if not time_ok():
                break

            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            worst = simplex[-1]

            # Reflection
            xr = centroid + alpha * (centroid - worst)
            xr = np.clip(xr, -5.0, 5.0)
            fr = safe(xr)
            if fr is None:
                break

            if fr < f_simplex[0]:
                # Expansion
                xe = centroid + gamma * (centroid - worst)
                xe = np.clip(xe, -5.0, 5.0)
                fe = safe(xe)
                if fe is None:
                    break
                if fe < fr:
                    simplex[-1] = xe
                    f_simplex[-1] = fe
                else:
                    simplex[-1] = xr
                    f_simplex[-1] = fr
            elif fr < f_simplex[-2]:
                simplex[-1] = xr
                f_simplex[-1] = fr
            else:
                # Contraction
                xc = centroid + rho * (worst - centroid)
                xc = np.clip(xc, -5.0, 5.0)
                fc = safe(xc)
                if fc is None:
                    break
                if fc < f_simplex[-1]:
                    simplex[-1] = xc
                    f_simplex[-1] = fc
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                        v = safe(simplex[i])
                        if v is None:
                            return np.array(simplex[0]), f_simplex[0]
                        f_simplex[i] = v

            # Re-sort
            order = np.argsort(f_simplex)
            simplex = [simplex[i] for i in order]
            f_simplex = [f_simplex[i] for i in order]

            # Update best
            if f_simplex[0] < best_f:
                best_f = f_simplex[0]
                x0 = simplex[0].copy()

        return x0, best_f

    def _cma_es(self, func, x0, n, budget, safe, time_ok, best_f):
        """CMA-ES optimizer."""
        rng = np.random.default_rng(1)
        lam = max(12, 4 + int(3 * np.log(n)))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / float(w @ w)
        c_sigma = (mu_eff + 2.0) / (n + mu_eff + 5.0)
        c_c = (4.0 + mu_eff / n) / (n + 4.0 + 2.0 * mu_eff / n)
        d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mu_eff - 1.0) / (n + 1.0)) - 1.0) + c_sigma
        c1 = 2.0 / (n ** 1.6 + 18.0)
        cmu = 2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((n + 2.0) ** 2 + mu_eff)
        chiN = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

        m = x0.copy()
        sigma = 1.0
        C = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)

        g = 0
        while budget > 0 and time_ok():
            if budget < lam:
                break

            Z = rng.standard_normal((lam, n))
            Y = (Z @ B.T) * D
            X = m[None, :] + sigma * Y
            X = np.clip(X, -5.0, 5.0)

            fs = []
            for x in X:
                v = safe(x)
                fs.append(1e18 if v is None else v)
            budget -= lam
            f = np.asarray(fs)

            fi = int(np.argmin(f))
            if f[fi] < best_f:
                best_f = float(f[fi])
                x0 = X[fi].copy()

            idx = np.argsort(f)
            top = idx[:mu]
            Ys = Y[top] / sigma
            Xs = X[top]

            ps = (1.0 - c_sigma) * ps + np.sqrt(c_sigma * (2.0 - c_sigma) * mu_eff) * (w @ Ys)
            hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - c_sigma) ** (2 * (g + 1)))) / chiN < \
                   1.4 + 2.0 / (n + 1)
            pc = (1.0 - c_c) * pc + hsig * np.sqrt(c_c * (2.0 - c_c) * mu_eff) * (w @ (Xs - m[None, :])) / sigma

            C = (1.0 - c1 - cmu) * C
            C += c1 * (np.outer(pc, pc) + (hsig - 1.0 + c1) * C)
            C += cmu * sum(wi * np.outer(y, y) for wi, y in zip(w, Ys))

            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / chiN - 1.0))
            sigma = float(np.clip(sigma, 1e-8, 10.0))

            m = w @ Xs
            m = np.clip(m, -5.0, 5.0)

            try:
                eig, B = np.linalg.eigh(C)
                eig = np.maximum(eig, 1e-12)
                D = np.sqrt(eig)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                B = np.eye(n)
                D = np.ones(n)
            g += 1

        return x0, best_f
