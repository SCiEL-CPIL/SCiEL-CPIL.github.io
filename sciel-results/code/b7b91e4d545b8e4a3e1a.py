"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (standard, with cumulative step-size adaptation) for BBOB f18."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start = time.time()
        deadline = start + 325.0  # safety margin under the 300s/330s limit

        n = self.dim
        budget = self.budget

        # Population size: 4 + 3 ln n  ->  7 for n=5
        lam = int(4 + 3 * np.log(n))
        if lam > budget // 5:
            lam = max(4, budget // 5)

        # Standard CMA-ES parameters
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = int(1.0 / w.sum())
        cc = (4.0 + mueff / n) / (n + 4.0 + 2.0 * mueff / n)
        cs = (mueff + 2.0) / (n + mueff + 5.0)
        c1 = (2.0 / ((n + 1.3) ** 2 + mueff)) * (3.0 - n) / (3.0 + n)
        cmu = min(
            1.0 - c1,
            2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0) ** 2 + mueff),
        )
        chi_n = np.sqrt(n) * (1.0 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n * n))

        # Initial mean: center of the box (robust to the hidden shift)
        m = np.zeros(n)
        sigma = 2.0
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)

        calls = 0

        def remaining():
            return budget - calls

        def can_continue():
            return calls < budget and (time.time() < deadline - 0.5)

        def evaluate(xs):
            """xs: (lam, n) array. Returns (lam,) values; updates calls."""
            nonlocal calls
            vals = np.empty(lam)
            for i in range(lam):
                if calls >= budget or time.time() > deadline:
                    break
                vals[i] = float(func(np.clip(xs[i], -5.0, 5.0)))
                calls += 1
            return vals

        # Initial population
        rng = np.random.default_rng(0)
        B = rng.standard_normal((lam, n))
        xs = m + sigma * B
        vals = evaluate(xs)
        order = np.argsort(vals)
        best_val = float(vals[order[0]])
        best_x = np.clip(xs[order[0]], -5.0, 5.0).copy()

        while can_continue():
            # Update mean from the mu best
            m_new = np.sum(w[:, None] * xs[order[:mu]], axis=0)

            # Update covariance matrix
            diff = (m_new - m) / sigma
            delta_s = np.linalg.norm(diff)
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * diff
            hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** 2)
                    / chi_n < 1.4 + 2.0 / (n + 1.0))
            pc = (1.0 - cc) * pc + hsig * cc * diff
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                di = (xs[order[i]] - m) / sigma
                rank_mu += w[i] * np.outer(di, di)
            C = ((1.0 - cs * c1 - cmu) * C
                 + cs * c1 * np.outer(pc, pc)
                 + cmu * rank_mu)
            m = m_new

            # Step-size adaptation
            sigma *= np.exp(
                cs / chi_n * (np.linalg.norm(ps) / chi_n - 1.0)
            )
            sigma = float(np.clip(sigma, 1e-12, 50.0))

            # New population
            B = rng.standard_normal((lam, n))
            xs = m + sigma * B
            vals = evaluate(xs)
            order = np.argsort(vals)

            cur = float(vals[order[0]])
            if cur < best_val:
                best_val = cur
                best_x = np.clip(xs[order[0]], -5.0, 5.0).copy()

            # Termination: stagnation (sigma tiny or mean barely moving)
            if sigma < 1e-10 or np.linalg.norm(m_new - m) < 1e-12:
                break

        return best_x, best_val
# EVOLVE-BLOCK-END
