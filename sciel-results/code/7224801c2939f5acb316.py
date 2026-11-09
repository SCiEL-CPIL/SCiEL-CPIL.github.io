# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES (population-based, full covariance) with a multi-start
    exploration phase, on [-5,5]^5, with a hard wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = time.monotonic() + 290.0  # safety margin vs 330s kill

    def _cma_step(self, x, C, D, ps, pc, sigma, rng, func, calls, max_calls):
        """One CMA-ES generation. Returns updated state and call count."""
        n = self.dim
        # lambda from Hansen (2016): 4 + floor(3 log(n))
        lam = int(4 + 3 * np.log(n))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        c_sigma = (mueff + 2) / ((n + 1) ** 2 + mueff)
        d_sigma = 1.0 + 2.0 * max(0.0, (mueff - 1) / (n + 1)) + c_sigma
        c_c = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
        c_c = min(c_c, 1.0 - c_sigma)
        c_1 = 2.0 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1.0 - c_1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2) ** 2 + mueff))
        chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

        B, Dvec = np.linalg.eigh(C)
        B = B[:, ::-1]
        Dvec = Dvec[::-1]
        Csq = Dvec * B

        for _ in range(max_calls - calls):
            if time.monotonic() > self.deadline:
                break
            # sample and evaluate
            ys = np.empty((lam, n))
            zs = np.empty((lam, n))
            for i in range(lam):
                z = rng.standard_normal(n)
                zs[i] = z
                ys[i] = x + sigma * (Csq @ z)
            fvals = np.empty(lam)
            for i in range(lam):
                fvals[i] = func(np.clip(ys[i], -5, 5))
                calls += 1
                if time.monotonic() > self.deadline:
                    break
            # sort by fitness
            order = np.argsort(fvals)[:mu]
            yw = ys[order]
            ymean = (w[:, None] * yw).sum(axis=0)
            # update mean
            x = ymean
            # step-size adaptation
            ps = (1.0 - c_sigma) * ps + (c_sigma * d_sigma / chi_n) * (Csq.T @ (B @ (Dvec * (B.T @ ymean))))
            sigma = sigma * np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1.0))
            sigma = max(sigma, 1e-12)
            # update covariance
            pc = (1.0 - c_c) * pc + c_c * (np.linalg.norm(ps) / chi_n < 1.4 + c_c) * ps
            delta_hs = (1.0 - (1.0 - c_c) ** 2)
            rank1 = np.outer(pc, pc)
            rankmu = np.zeros((n, n))
            for i in range(mu):
                zi = B @ (Dvec * (B.T @ (yw[i] - ymean)))
                rankmu += w[i] * np.outer(zi, zi)
            C = (1.0 + c_1 * delta_hs + c_mu) * C - c_1 * rank1 - c_mu * rankmu
            # ensure positive definite
            C = 0.5 * (C + C.T)
            # re-derive B, Dvec
            try:
                B, Dvec = np.linalg.eigh(C)
            except np.linalg.LinAlgError:
                B, Dvec = np.eye(n), np.ones(n)
            B = B[:, ::-1]
            Dvec = Dvec[::-1]
            Csq = Dvec * B
            if max_calls - calls <= 0:
                break
        return x, C, D, ps, pc, sigma, calls

    def __call__(self, func):
        rng = np.random.default_rng(0)
        n = self.dim
        budget = self.budget
        calls = 0
        # ---- Phase 1: multi-start exploration to find a promising region ----
        n_starts = min(8, budget // 10)
        best_x = None
        best_f = np.inf
        for s in range(n_starts):
            x0 = rng.uniform(-5, 5, n)
            f0 = func(x0)
            calls += 1
            if f0 < best_f:
                best_f, best_x = f0, x0.copy()
            if time.monotonic() > self.deadline:
                break
        # ---- Phase 2: CMA-ES refinement from best start ----
        x = best_x.copy()
        sigma = 1.5
        C = np.eye(n)
        ps = np.zeros(n)
        pc = np.zeros(n)
        calls, _ = self._cma_step(x, C, None, ps, pc, sigma, rng, func, calls, budget)
        return x, best_f
# EVOLVE-BLOCK-END
