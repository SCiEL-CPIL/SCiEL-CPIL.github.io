"""Derivative-free multi-start optimizer with CMA-ES basin exploitation."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start: random/Halton global search -> Brent basin ID -> CMA-ES -> L-BFGS polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    # ---------- helpers ----------
    def _halton(self, n, d, base, seed):
        """Van der Corput / Halton sequence in [0,1)^d, starting at base."""
        pts = np.zeros((n, d))
        for i in range(n):
            k = base + i
            row = np.empty(d)
            for j in range(d):
                f, r = 1.0, float(k)
                while r >= 1.0:
                    r /= (j + 2)
                    f *= (j + 2)
                row[j] = r / f
            pts[i] = row
        # cheap deterministic shuffle to decorrelate
        perm = (np.arange(n) * (seed + 17)) % n
        return pts[perm]

    def _brent(self, func, x0, fx, budget, deadline):
        """Brent's direct search, box-constrained."""
        n = self.dim
        best_x, best_f = x0.copy(), fx
        step = 1.0
        while budget > 0 and time.time() < deadline:
            improved = False
            for i in range(n):
                if budget <= 0 or time.time() >= deadline:
                    break
                for s in (1.0, -1.0):
                    if budget <= 0 or time.time() >= deadline:
                        break
                    cand = best_x.copy()
                    cand[i] = np.clip(cand[i] + s * step, -5.0, 5.0)
                    cf = func(cand)
                    budget -= 1
                    if cf < best_f:
                        best_x, best_f, improved = cand, cf, True
            if improved:
                step *= 1.618
            else:
                step *= 0.5
                if step < 1e-6:
                    break
        return best_x, best_f, budget

    def _cma(self, func, x0, fx, budget, deadline, sigma0, maxiter):
        """CMA-ES with box constraints (reflect), adaptive sigma, mean-average restart."""
        n = self.dim
        x = x0.copy()
        fbest = fx
        lam = int(4 + 3 * np.log(n))
        C = np.eye(n)
        sigma = sigma0
        pc = np.zeros(n)
        ps = np.zeros(n)
        c1 = (2 / ((n + 1) ** 2 + 10)) / lam
        cmu = min(1 - c1,
                  2 * ((n + 1.5) ** 2 - 2) / ((n + 3) ** 2 + 6)) / lam
        cs = (n + 1.7 - 1 / (n + 2.2)) / (n + n / 2 + 1.7)
        damps = 1 + 2 * max(0, np.sqrt((lam / 2 - 1) / (n + 1)) - 1) + cs
        c_sigma = cs / damps
        c_c = (4 + cmu / c1) / (n + 4)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        it = 0
        while budget > 0 and it < maxiter and time.time() < deadline:
            it += 1
            # sample population
            B = np.linalg.cholesky(C + 1e-20 * np.eye(n))
            W = np.zeros(n)
            for _ in range(n):
                W = W + B @ (B.T @ np.random.randn(n))
            W /= np.linalg.norm(W)
            D = np.sqrt(np.einsum('ij,ij->i', C, B))
            pop = np.empty((lam, n))
            for k in range(lam):
                z = np.random.randn(n)
                pop[k] = x + sigma * (D * (B.T @ z) + 1e-10)
            pop = np.clip(pop, -5.0, 5.0)
            fvals = np.empty(lam)
            for k in range(lam):
                if budget <= 0 or time.time() >= deadline:
                    fvals[k] = np.inf
                    continue
                fvals[k] = func(pop[k])
                budget -= 1
            order = np.argsort(fvals)
            m = lam // 2
            yw = np.zeros(n)
            mu_eff = 1.0 / np.sum(1.0 / np.arange(1, m + 1))
            for k in range(m):
                yw += (1.0 / (k + 1)) * pop[order[k]]
            yw *= (1.0 / mu_eff)
            xnew = x + yw
            xnew = np.clip(xnew, -5.0, 5.0)
            # update statistics
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * lam) * (C @ (yw / sigma))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (it + 1))) < (1.4 + 2 / (n + 1)) * chi_n
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * lam) * yw
            delta_h = (1 - hsig) * c_c * (2 - c_c)
            C = (1 + c1 * delta_h - c1 - cmu) * C \
                + c1 * np.outer(pc, pc) \
                + cmu * sum((1.0 / (k + 1)) * np.outer(pop[order[k]] - x, pop[order[k]] - x) for k in range(m))
            C = (C + C.T) / 2.0
            x = xnew
            if fvals[order[0]] < fbest:
                fbest = fvals[order[0]]
            # sigma adaptation
            sigma *= np.exp((c_sigma / damps) * (np.linalg.norm(ps) / chi_n - 1.0))
            sigma = min(max(sigma, 1e-12), 10.0)
            # mean-average restart
            if it > 30:
                mean_f = np.mean(fvals[order[:m]])
                if mean_f < fbest * 0.9 and fbest > 0:
                    x = pop[order[0]].copy()
                    fbest = fvals[order[0]]
                    C = np.eye(n)
                    pc = ps = np.zeros(n)
                    sigma = sigma0
                    it = 0
        return x, fbest, budget

    # ---------- main ----------
    def __call__(self, func):
        n = self.dim
        deadline = time.time() + 285.0  # safety margin under 300s
        budget = self.budget
        best_x = None
        best_f = np.inf

        # --- Phase 1: global multi-start (random + Halton) ---
        starts = []
        # deterministic random
        rng = np.random.default_rng(0)
        for _ in range(40):
            starts.append(rng.uniform(-5.0, 5.0, n))
        # Halton quasi-random
        for _ in range(40):
            starts.append(self._halton(1, n, 0, 7)[0] * 10.0 - 5.0)
        # center
        starts.append(np.zeros(n))
        starts = np.array(starts)
        starts = np.clip(starts, -5.0, 5.0)

        # evaluate all starts
        fstarts = np.empty(len(starts))
        for i, s in enumerate(starts):
            if budget <= 0 or time.time() >= deadline:
                fstarts[i:] = np.inf
                break
            fstarts[i] = func(s)
            budget -= 1
        order = np.argsort(fstarts)
        best_f = fstarts[order[0]]
        best_x = starts[order[0]].copy()

        # --- Phase 2: basin identification via Brent on top starts ---
        for idx in order:
            if budget < 30 or time.time() >= deadline:
                break
            x0 = starts[idx].copy()
            fx0 = fstarts[idx]
            # quick local check: if already good, skip heavy Brent
            x1, f1, budget = self._brent(func, x0, fx0, min(budget, 60), deadline)
            if f1 < best_f:
                best_f, best_x = f1, x1.copy()
            if best_f < 1e-6:
                break

        # --- Phase 3: CMA-ES basin exploitation ---
        if budget > 30 and time.time() < deadline:
            x2, f2, budget = self._cma(func, best_x, best_f, budget, deadline,
                                       sigma0=1.0, maxiter=200)
            if f2 < best_f:
                best_f, best_x = f2, x2.copy()

        # --- Phase 4: polish with L-BFGS-B (gradient-based, box) ---
        if budget > 5 and time.time() < deadline:
            try:
                res = minimize(func, best_x, method='L-BFGS-B',
                               bounds=[(-5.0, 5.0)] * n,
                               options={'maxiter': 200, 'maxfun': 50,
                                        'gtol': 1e-8, 'ftol': 1e-12})
                if res.fun < best_f:
                    best_f, best_x = res.fun, np.clip(res.x, -5.0, 5.0)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
