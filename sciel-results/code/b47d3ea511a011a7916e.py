"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer for BBOB f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Population size
        lam = max(4, int(4 + 3 * np.log(n)))
        # CMA-ES parameters
        cm = 1 / (n + 4)
        cs = (n + 1.3) / (n + 3 + 2 * n / n)
        ds = 1 + 2 * max(0, np.sqrt((n - 1) / (n + 1)) - 1) + cs
        cc = (4 + n / n) / (4 + n + n / n)
        c1 = (2 + 2 / (n + 1.3)) / ((n + 2) ** 2 + 2)
        cmu = min(c1 * 2, (2 - c1 + 2 / (n + 1.3)) / ((n + 2) ** 2 + 2))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Initial state
        x = rng.uniform(-5, 5, n)
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        xmean = x.copy()
        fmean = func(xmean)
        best_x = xmean.copy()
        best_f = fmean

        calls = 1

        while calls < budget:
            # Generate population
            offspring = []
            offspring_f = []
            for _ in range(lam):
                z = rng.standard_normal(n)
                y = xmean + sigma * (C @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                offspring.append(y)
                offspring_f.append(fy)
                if calls >= budget:
                    break

            # Sort by fitness
            offspring_f = np.array(offspring_f)
            offspring = np.array(offspring)
            order = np.argsort(offspring_f)
            offspring = offspring[order]
            offspring_f = offspring_f[order]

            # Update best
            if offspring_f[0] < best_f:
                best_f = offspring_f[0]
                best_x = offspring[0].copy()

            if calls >= budget:
                break

            # Update mean
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1 / np.sum(w ** 2)
            xmean_old = xmean.copy()
            xmean = np.sum(w[:, None] * offspring[:mu], axis=0)

            # Update path covariance adaptation
            pc = (1 - cs) * pc + cs * np.sqrt(mueff) / np.linalg.norm(np.sqrt(C)) @ (xmean - xmean_old) / sigma
            hsig = (np.linalg.norm(pc) / np.sqrt(1 - (1 - cs) ** (calls + 1))) / chi_n < 1.4 + 2 / (n + 1)
            ps = (1 - ds) * ps + ds * hsig * np.sqrt(mueff) / np.linalg.norm(np.sqrt(C)) @ (xmean - xmean_old) / sigma

            # Update C
            rank_one = np.outer(ps, ps)
            if hsig:
                rank_mu = np.sum(w[:, None, None] * np.outer(offspring[:mu] - xmean_old, offspring[:mu] - xmean_old), axis=0)
                rank_mu *= 1 / sigma ** 2
            else:
                rank_mu = np.zeros((n, n))
            C = (1 + c1 * cmu - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu
            C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100)

            # Ensure C is positive definite
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-10:
                C += (1e-10 - np.min(eigvals) + 1e-8) * np.eye(n)

        return best_x, best_f
# EVOLVE-BLOCK-END
