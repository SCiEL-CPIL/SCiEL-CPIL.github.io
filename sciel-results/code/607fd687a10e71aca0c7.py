"""Fresh design for the suite task: CMA-ES with multi-start on BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np


class Optimizer:
    """Multi-start CMA-ES minimizer for a box-constrained black-box function."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def _cma(self, func, rng, x0, f0, iters, deadline):
        """CMA-ES from start point x0 (with known value f0) for up to `iters` steps.

        Returns (best_x, best_f, calls_made, stopped_by_deadline).
        """
        n = self.dim
        x = np.array(x0, dtype=float)
        f = float(f0)
        best_x = x.copy()
        best_f = f
        calls = 1  # f0 already consumed

        # population size
        m = max(4, int(math.ceil(4 + 3 * math.log(n))))
        # sigma: start relative to box, scaled by dim
        sigma = 2.0 / math.sqrt(n)

        # weights
        mu = m // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, m + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w[:mu] ** 2)
        c_sigma = (mueff + 2.0) / (n + mueff + 4.0)
        c_c = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
        c_1 = 2.0 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1.0 - c_1,
                   (2.0 * (mueff - 2.0 + 2.0 / n)) /
                   ((n + 2.0) ** 2 + mueff))
        d_sigma = 1.0 + 2.0 * max(0.0, math.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) \
            + c_sigma

        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = math.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n ** 2))

        # sqrt of C via Cholesky; if it fails, use eigendecomposition fallback
        def chol_or_eig(M):
            try:
                return np.linalg.cholesky(M)
            except np.linalg.LinAlgError:
                vals, vecs = np.linalg.eigh(M)
                vals = np.clip(vals, 0.0, None)
                return vecs @ np.diag(np.sqrt(vals))

        B = np.eye(n)
        D = np.ones(n)

        for _ in range(iters):
            if time.time() >= deadline:
                return best_x, best_f, calls, True

            # sample
            ys = np.array([B @ (sigma * (D * rng.standard_normal(n))) for _ in range(m)])
            pops = np.clip(x[None, :] + ys, -5.0, 5.0)
            fs = np.array([func(pops[i]) for i in range(m)])
            calls += m

            order = np.argsort(fs)
            ys_sorted = ys[order]
            fs_sorted = fs[order]

            # update best
            if fs_sorted[0] < best_f:
                best_f = float(fs_sorted[0])
                best_x = pops[order[0]].copy()

            # mean update
            yw = np.sum(w[:mu, None] * ys_sorted[:mu], axis=0)
            x = np.clip(x + yw, -5.0, 5.0)

            # evolution paths
            ps = (1.0 - c_sigma) * ps + math.sqrt(c_sigma * (2.0 - c_sigma) * mueff) * (B @ (D * yw))
            hsig = (np.linalg.norm(ps) / math.sqrt(1.0 - (1.0 - c_sigma) ** 2) / chi_n) < (1.4 + 2.0 / (n + 1.0))
            pc = (1.0 - c_c) * pc + hsig * math.sqrt(c_c * (2.0 - c_c) * mueff) * yw

            # covariance update
            delta_h = (1.0 - hsig) * c_c * (2.0 - c_c)
            rank1 = np.outer(pc, pc)
            rankmu = np.sum(w[:mu, None, None] * np.outer(ys_sorted[:mu], ys_sorted[:mu]), axis=0)
            C = (1.0 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank1 + c_mu * rankmu

            # sigma update
            sigma = sigma * math.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1.0))
            sigma = min(sigma, 1e3)

            # recompute sqrt(C)
            B, D = self._sqrt_cov(C)
            C = 0.5 * (C + C.T)  # symmetrize

        return best_x, best_f, calls, False

    @staticmethod
    def _sqrt_cov(C):
        try:
            L = np.linalg.cholesky(C + 1e-12 * np.eye(C.shape[0]))
            return L, np.ones(C.shape[0])
        except np.linalg.LinAlgError:
            vals, vecs = np.linalg.eigh(C)
            vals = np.clip(vals, 0.0, None)
            D = np.sqrt(vals)
            B = vecs
            return B, D

    def __call__(self, func):
        start = time.time()
        deadline = start + 280.0  # safety margin under 330s kill
        n = self.dim
        budget = self.budget

        # ---- Phase 1: cheap random multi-start to find a good basin ----
        rng = np.random.default_rng(12345)
        # how many cheap random points? keep small to preserve budget for CMA
        n_random = min(60, budget // 8)
        best_x = None
        best_f = np.inf
        for _ in range(n_random):
            if time.time() >= deadline:
                break
            cand = rng.uniform(-5.0, 5.0, n)
            fc = func(cand)
            if fc < best_f:
                best_f = fc
                best_x = cand.copy()

        # ---- Phase 2: CMA-ES from the best random point ----
        remaining = budget - n_random
        if remaining > 0 and best_x is not None:
            bx, bf, calls, _ = self._cma(func, rng, best_x, best_f, remaining, deadline)
            if bf < best_f:
                best_f = bf
                best_x = bx

        # ---- Phase 3: if budget remains, a second CMA from a fresh random start ----
        used = n_random + (remaining if remaining > 0 else 0)
        if used < budget and time.time() < deadline:
            cand = rng.uniform(-5.0, 5.0, n)
            fc = func(cand)
            if fc < best_f:
                best_f = fc
                best_x = cand.copy()
            leftover = budget - used - 1
            if leftover > 0:
                bx, bf, calls, _ = self._cma(func, rng, best_x, best_f, leftover, deadline)
                if bf < best_f:
                    best_f = bf
                    best_x = bx

        return best_x, best_f
# EVOLVE-BLOCK-END
