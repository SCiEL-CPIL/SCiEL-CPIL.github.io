"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Lattice-seeded CMA-ES for Rastrigin rotated."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _cmaes(self, func, x0, fx0, budget, sigma0, deadline):
        # CMA-ES with adaptive sigma (CSA-style 1/5 rule)
        n = self.dim
        c1 = (2 + 2.0 / (n + 1.3)) / ((n + 2) ** 2 + 2)
        cmu = min(1 - c1,
                  2 * (2 + 2.0 / (n + 1.3)) / ((n + 2) ** 2 + 2))
        # population size
        lam = max(4, int(4 + 3 * np.log(n)))
        # step-size adaptation
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n * n))
        d_sigma = 1 + 2 * max(0, np.sqrt((lam + 1) / (n + 1)) - 1) + c_sigma
        c_sigma = (3 + lam / 2) / (n + 9)
        c_c = (4 + lam / n) / (n + 4)
        c1 = (2 + 2.0 / (n + 1.3)) / ((n + 2) ** 2 + 2)
        cmu = min(1 - c1,
                  2 * (2 + 2.0 / (n + 1.3)) / ((n + 2) ** 2 + 2))
        # state
        x = np.asarray(x0, dtype=float).copy()
        C = np.eye(n)
        Csqrt_inv = np.eye(n)  # we use B^-1 D^-1 via cholesky each step
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        sigma = sigma0
        best_x = x.copy()
        best_fx = fx0
        calls = 1
        mu = lam // 2
        w = np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / (w ** 2).sum()
        for _ in range(budget):
            if time.time() > deadline:
                break
            # sample
            ys = np.zeros((lam, n))
            fvals = np.zeros(lam)
            for i in range(lam):
                z = np.random.standard_normal(n)
                y = x + sigma * (Csqrt_inv @ z)
                y = np.clip(y, -5, 5)
                ys[i] = y
                fvals[i] = func(y)
                calls += 1
                if fvals[i] < best_fx:
                    best_fx = fvals[i]
                    best_x = y.copy()
            # sort
            order = np.argsort(fvals)
            ys = ys[order]
            fvals = fvals[order]
            # recompute Csqrt_inv via cholesky of C
            try:
                L = np.linalg.cholesky(C)
                Csqrt_inv = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                L = np.linalg.cholesky(C + 1e-8 * np.eye(n))
                Csqrt_inv = np.linalg.inv(L)
            # update
            m_new = (w @ ys[:mu]).reshape(n)
            ym = (m_new - x) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * ym
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (n + 1)) * chi_n)
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * ym
            rank_c = np.outer(ym, ym)
            if hsig:
                rank_mu = np.zeros((n, n))
                for i in range(mu):
                    ymi = (ys[i] - x) / sigma
                    rank_mu += w[i] * np.outer(ymi, ymi)
                C = (1 + c1 * cmu - c1 - cmu) * C + c1 * rank_c + cmu * rank_mu
            else:
                C = (1 + c1 * cmu - c1) * C + c1 * rank_c
            # sigma adaptation
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = min(sigma, 1e6)
            x = m_new
            x = np.clip(x, -5, 5)
            if time.time() > deadline:
                break
        return best_x, best_fx, calls

    def __call__(self, func):
        deadline = time.time() + 320.0  # safety margin
        n = self.dim
        # 1) lattice of integer candidates (Rastrigin minima at integers)
        # integers in [-5,5] -> -5..5, but clip; use -4..4 to stay inside box
        lattice = []
        for a in range(-4, 5):
            for b in range(-4, 5):
                for c in range(-4, 5):
                    for d in range(-4, 5):
                        for e in range(-4, 5):
                            lattice.append([a, b, c, d, e])
        lattice = np.array(lattice, dtype=float)
        # evaluate lattice (this is 9^5 = 59049, too many for budget 500)
        # instead: random subset + a few structured points
        # Use a coarse grid: integers in {-4,-3,-2,-1,0,1,2,3,4} is 9^5=59049 > 500
        # So sample a random subset of the lattice
        rng = np.random.default_rng(0)
        # evaluate up to ~150 lattice points, prioritizing spread
        idx = rng.permutation(len(lattice))[:150]
        best_x = None
        best_fx = np.inf
        for i in idx:
            if time.time() > deadline:
                break
            x = lattice[i]
            fx = func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
        # also a couple random continuous points for safety
        for _ in range(5):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
        calls = 155
        # 2) CMA-ES refinement around best candidate
        remaining = self.budget - calls
        if remaining > 10 and best_x is not None:
            bx, bfx, used = self._cmaes(func, best_x, best_fx, remaining, 0.8, deadline)
            if bfx < best_fx:
                best_fx = bfx
                best_x = bx
        return best_x, best_fx
# EVOLVE-BLOCK-END
