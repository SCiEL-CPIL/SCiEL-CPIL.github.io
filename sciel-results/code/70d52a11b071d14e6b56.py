"""Rewritten optimizer for BBOB f14 (different powers)."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-strategy black-box optimizer with explicit wall-clock deadline."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _run(self, func):
        dim = self.dim
        start = time.time()
        deadline = start + 325.0  # safety margin before 330 s kill
        calls = 0

        def call(x):
            nonlocal calls
            calls += 1
            return func(np.clip(np.asarray(x, dtype=float), -5.0, 5.0))

        # ---- Phase 1: global random sampling to find a good basin ----
        rng = np.random.default_rng(12345)
        best_x = None
        best_f = np.inf
        n_global = 40
        for i in range(n_global):
            x = rng.uniform(-5.0, 5.0, dim)
            f = call(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            if time.time() > deadline - 10.0:
                break

        # ---- Phase 2: CMA-ES refinement from the best global point ----
        # CMA-ES parameters (dim=5)
        pop = 4 + int(3 * np.log(dim))  # 13
        sigma = 2.0
        cc = (4 + dim / pop) / (dim + 4 + 2.0 / pop)
        cs = (pop / 2 + 1) / (dim + pop / 2 + 3)
        c1 = 2 / ((dim + 1.3) ** 2 + pop)
        cmua = min(1 - c1, 2 / ((dim + 2) ** 2 + pop))
        ds = 1 + 2 * max(0, np.sqrt((pop / 2 + 1) / (dim + 1)) - 1) + cs

        xmean = best_x.copy()
        fmean = best_f
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        step_success = 0.0

        def update_sigma():
            nonlocal sigma, step_success
            # adaptive sigma: push sigma toward the successful step size
            target = (1.0 + 2.0 * (step_success / (1 + calls) - 1 / (pop + 1)))
            # log-linear adaptation
            sigma *= np.exp(cs / ds * (np.log(np.linalg.norm(ps)) / chiN - 1.0))
            sigma = np.clip(sigma, 1e-6, 50.0)

        while calls < self.budget and time.time() < deadline - 5.0:
            # generate population
            W = np.zeros((pop, dim))
            F = np.zeros(pop)
            for i in range(pop):
                wi = 1.0 / np.sqrt(2 * (i + 1) * (pop - i))
                y = B @ (D * (wi * rng.standard_normal(dim)))
                x = np.clip(xmean + sigma * y, -5.0, 5.0)
                W[i] = y
                F[i] = call(x)
                if time.time() > deadline - 5.0:
                    break
            if calls >= self.budget:
                break
            # sort by fitness
            order = np.argsort(F)
            mu = pop // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1 / np.sum(weights ** 2)
            yw = np.sum(weights[:, None] * W[order[:mu]], axis=0)
            # update mean
            xmean_new = np.clip(xmean + sigma * yw, -5.0, 5.0)
            # update pc
            pc = (1 - cc) * pc + cc * (1 / np.sqrt(cc * (2 - cc))) * yw
            norm_pc = np.linalg.norm(pc)
            hsig = (norm_pc / np.sqrt(1 - (1 - cc) ** (2 * (calls + 1) / (pop + 1)))) / chiN - (1 - cc)
            hsig = 1.0 if hsig > 1.4 else 0.0
            # update ps
            ps = (1 - cs) * ps + hsig * np.sqrt(cs * (2 - cs) * (1 - (1 - cs) ** 2)) * yw
            # update covariance matrix
            delta_h = (1 - hsig) * cc * (2 - cc)
            C_update = np.outer(pc, pc) * delta_h
            outer_yw = np.outer(yw, yw)
            C_update += (np.sum(weights[:, None] * np.outer(W[order[:mu]], W[order[:mu]]), axis=0)
                         * c1 * mueff / (mueff * 1))  # simplified rank-mu
            # Use standard rank-mu
            C_update = np.outer(pc, pc) * delta_h
            C_update += c1 * (np.outer(yw, yw) + (1 - hsig * (1 - cc)) * 0)
            # proper rank-mu term
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(W[order[i]], W[order[i]])
            C_update += c1 * (1 - hsig * (1 - cc)) * np.outer(yw, yw) + c1 * (mueff * 0)
            # Combine correctly
            C_new = (1 + c1 * (mueff - 2) + delta_h - c1) * (B @ (D ** 2) @ B.T)
            C_new += c1 * (np.outer(yw, yw) * (1 - hsig * (1 - cc)) + rank_mu) + np.outer(pc, pc) * delta_h
            # Re-orthogonalize (eigh)
            try:
                D2, B2 = np.linalg.eigh(C_new)
                D2 = np.sqrt(np.clip(D2, 1e-12, None))
                B = B2
                D = D2
            except np.linalg.LinAlgError:
                pass
            # update sigma
            update_sigma()
            # track best
            xmean = xmean_new
            for i in range(pop):
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = np.clip(xmean + sigma * W[i], -5.0, 5.0)
            step_success = float(np.sum(F[order[:mu]] < fmean))
            fmean = float(np.min(F))

        return best_x, best_f

    def __call__(self, func):
        try:
            return self._run(func)
        except Exception:
            # fallback: return a sensible point
            return np.zeros(self.dim), np.inf
# EVOLVE-BLOCK-END
