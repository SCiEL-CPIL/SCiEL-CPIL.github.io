"""Fresh design: multi-start CMA-ES + Nelder-Mead polish with a wall-clock deadline guard."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-start CMA-ES (small lambda) with Nelder-Mead polish and a time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 275.0          # safety margin well under the 300 s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        def tleft():
            return deadline - time.time()

        # Counted wrapper: every func evaluation is recorded and time-checked.
        calls = 0

        def feval(x):
            nonlocal calls
            if calls >= budget:
                return float('inf')
            calls += 1
            try:
                v = func(np.asarray(x, dtype=float))
            except Exception:
                return float('inf')
            if not np.isfinite(v):
                return float('inf')
            return float(v)

        best_x = None
        best_f = float('inf')

        def track(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = np.array(x, dtype=float)
                return True
            return False

        # ---------- 1. Multi-start exploration ----------
        n_starts = 15
        start_pts = [np.clip(rng.uniform(-5, 5, dim), -5, 5) for _ in range(n_starts)]
        start_vals = [feval(p) for p in start_pts]
        for p, v in zip(start_pts, start_vals):
            track(p, v)

        # ---------- 2. CMA-ES exploitation around each start ----------
        lam = 10
        pop = np.zeros((lam, dim))
        weights = np.zeros(lam)
        wsum = np.exp(-5.0 * np.arange(lam) / (lam - 1))
        weights = wsum / wsum.sum()
        mueff = lam / (4.0 + 2.0 * (dim / lam - 1.0))
        cmu = (2.0 - mueff / lam) / ((lam + 2.0) **2 - 1.0)
        ccs = (mueff + 1.5) / (lam + mueff + 5.0)
        cc = (4.0 + mueff / lam) / (lam + 4.0 + 2.0 * mueff / lam)
        cs = (mueff + 1.0) / (lam + mueff + 3.0)
        c1 = (2.0 - mueff / lam + cmu) / ((dim + 1.3) **2 - 2.0)
        cmu = (2.0 * (mueff / lam - 1.0) + cmu) / ((dim + 1.3) **2 - 2.0 + 2.0 * (mueff / lam - 1.0) + cmu)
        damps = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs
        # (c1 and cmu are computed above; keep them for the update loop)

        for idx in range(n_starts):
            if tleft() <= 0.0:
                break
            x = start_pts[idx]
            fx = start_vals[idx]
            if not np.isfinite(fx):
                continue
            sigma = 0.8
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            for _ in range(lam):
                z = rng.standard_normal((lam, dim))
                L = np.linalg.cholesky(C)
                pop = x[None, :] + sigma * (z @ L.T)
                pop = np.clip(pop, -5, 5)
                fvals = [feval(pop[i]) for i in range(lam)]
                order = np.argsort(fvals)
                for i in range(lam):
                    track(pop[i], fvals[i])
                if tleft() <= 0.0:
                    return best_x, best_f
                m = np.sum(weights[:, None] * pop[order], axis=0)
                ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) / sigma * (m - x)
                if np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** 2) < (1.4 + cc) * np.sqrt(lam + 1.0 - 2.0):
                    pc = (1.0 - ccs) * pc + ccs * (m - x) / sigma
                C = (1.0 + c1 + cmu - c1 - 1.0) * C  # placeholder (corrected below)
                # covariance update
                C = (1.0 - c1 - cmu) * C + c1 * np.outer(pc, pc)
                C += cmu * np.sum(weights[:int(mueff)] * np.outer(pop[order[:int(mueff)] - m],
                                                                  pop[order[:int(mueff)] - m]), axis=0)
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** 2) - 1.0))
                x = m
                fx = fvals[order[0]]
                if not np.isfinite(fx):
                    break
                if sigma < 1e-7:
                    break
            # keep best_x tracked throughout

        # ---------- 3. Nelder-Mead polish on the best point ----------
        if best_x is not None and tleft() > 0.0:
            best_x = self._nelder_mead(func, feval, best_x, tleft)
            track(best_x, feval(best_x) if False else best_f)  # best_f already tracked

        if best_x is None:
            best_x = np.zeros(dim)
        return best_x, best_f

    def _nelder_mead(self, func, feval, x0, tleft):
        dim = self.dim
        x0 = np.array(x0, dtype=float)
        # simplex
        simplex = [x0.copy() for _ in range(dim + 1)]
        for i in range(dim):
            simplex[i + 1] = x0.copy()
            simplex[i + 1][i] += 0.5 if x0[i] < 4.5 else -0.5
        fvals = [feval(s) for s in simplex]
        alpha, gamma, rho = 1.0, 2.0, 0.5
        while tleft() > 0.0:
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = [fvals[i] for i in order]
            centroid = np.mean(simplex[:-1], axis=0)
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = feval(xr)
            if fr < fvals[0]:
                xe = centroid + gamma * (xr - centroid)
                xe = np.clip(xe, -5, 5)
                fe = feval(xe)
                if fe < fr:
                    simplex[-1], fvals[-1] = xe, fe
                else:
                    simplex[-1], fvals[-1] = xr, fr
            elif fr < fvals[-2]:
                simplex[-1], fvals[-1] = xr, fr
            else:
                xc = centroid + rho * (simplex[-1] - centroid)
                xc = np.clip(xc, -5, 5)
                fc = feval(xc)
                if fc < fvals[-1]:
                    simplex[-1], fvals[-1] = xc, fc
                else:
                    for i in range(1, dim + 1):
                        simplex[i] = 0.5 * (simplex[i] + simplex[0])
                        fvals[i] = feval(simplex[i])
            if max(fvals) - min(fvals) < 1e-10:
                break
        return simplex[0]
# EVOLVE-BLOCK-END
