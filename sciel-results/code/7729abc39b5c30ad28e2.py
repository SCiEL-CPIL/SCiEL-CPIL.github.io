"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES with deadline guard for f21 (multi-modal)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes(self, func, x0, f0, sigma, budget, deadline, rng):
        dim = self.dim
        n = max(4, int(2 + 3 * np.log(dim)))
        c = 1.0 / (dim + 4.0)
        damps = 0.5 + 1.0 / (dim + 0.3) - c / 2.0
        cmu = min(2.0 - damps, 1.0 / (dim + 3.0))
        c_sigma = (1 + 2 * max(0.0, np.sqrt((dim - 1) / (dim + 3)) - 1)) / (dim + 6)
        d_sigma = 1 + 2 * max(0.0, (dim - 1) / (dim + 3) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        C_12 = np.sqrt(1.0 / (2.0 * dim)) * np.eye(dim)
        C_12 += 1.0 / (2 * dim)

        ps = 0.0
        s = 0.0
        s2 = 0.0
        count = 0
        best_x = x0.copy()
        best_f = f0

        for _ in range(budget):
            if time.time() > deadline:
                break
            # generate candidates
            ys = []
            for _ in range(n):
                z = rng.standard_normal(dim)
                y = x0 + sigma * (C_12 @ z)
                ys.append(y)
            ys = np.array(ys)
            fvals = np.array([func(y) for y in ys])

            # update best
            idx_min = int(np.argmin(fvals))
            if fvals[idx_min] < best_f:
                best_f = fvals[idx_min]
                best_x = ys[idx_min].copy()

            # sort
            order = np.argsort(fvals)
            ys_sorted = ys[order]
            fvals_sorted = fvals[order]
            mu = n // 2
            w = np.log(mu + 0.5) - np.log(1 + np.arange(mu))
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)

            m_new = np.sum(w[:, None] * ys_sorted[:mu], axis=0)
            diff = m_new - x0
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * diff
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (count + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
            p_c = (1 - c) * p_c + hsig * np.sqrt(c * (2 - c) * mu_eff) * diff / sigma
            rank_mu = np.outer(ys_sorted[:mu] - m_new, ys_sorted[:mu] - m_new)
            C = (1 + c_sigma * (2 - c_sigma) - c) * C \
                + c * (np.outer(p_c, p_c) + (1 - hsig) * c_sigma * (2 - c_sigma) * np.outer(p_sigma, p_sigma)) \
                + cmu * rank_mu
            # recompute C_12
            C_12 = np.sqrt(1.0 / (2.0 * dim)) * np.eye(dim)
            C_12 += 1.0 / (2 * dim)
            # symmetric sqrt via eigh for stability
            try:
                evals, evecs = np.linalg.eigh(C)
                evals = np.clip(evals, 1e-12, None)
                C_12 = evecs @ np.diag(np.sqrt(evals)) @ evecs.T
            except np.linalg.LinAlgError:
                C_12 = np.eye(dim)

            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * np.linalg.norm(diff) / sigma
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = float(np.clip(sigma, 1e-12, 5.0))

            x0 = m_new
            count += 1
        return best_x, best_f

    def __call__(self, func):
        start = time.time()
        deadline = start + 320.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        # initial population: deterministic grid + random
        n_init = 20
        xs = [np.array([0.0] * dim)]
        # corners / spread
        for i in range(1, n_init):
            xs.append(rng.uniform(-4.5, 4.5, dim))
        fvals = [func(x) for x in xs]

        best_i = int(np.argmin(fvals))
        best_x = xs[best_i].copy()
        best_f = fvals[best_i]
        used = n_init
        remaining = budget - used

        # allocate remaining budget among a few best starts
        k = 3
        order = np.argsort(fvals)
        starts = [(xs[i], fvals[i]) for i in order[:k]]

        per = remaining // k
        for i, (sx, sf) in enumerate(starts):
            rem_budget = per if i < k - 1 else remaining - per * i
            if rem_budget <= 0 or time.time() > deadline:
                break
            sigma = 1.5
            bx, bf = self._cmaes(func, sx.copy(), sf, sigma, rem_budget, deadline, rng)
            if bf < best_f:
                best_f = bf
                best_x = bx.copy()
            used += rem_budget

        return best_x, best_f
# EVOLVE-BLOCK-END
