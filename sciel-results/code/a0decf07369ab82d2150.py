"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style optimizer with population and adaptive step size."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initial population size
        lam = max(4, int(4 + 3 * np.log(d)))
        # Adaptive step size
        sigma = 1.0
        # Mean
        m = np.zeros(d)
        # Initial covariance matrix (identity)
        C = np.eye(d)
        # Precompute decomposition
        B, D = np.linalg.eigh(C)
        # Learning rates
        cmu = 1 / (d + 4)
        cc = (4 + d / 2) / (d + 4 + 3)
        c1 = 2 / ((d + 1.3) ** 2 + d)
        cs = (mu + 1) / (d + mu + 5)
        ds = 1 + 2 * max(0, np.sqrt((mu - 1) / (d + 1)) - 1) + cs
        mu = lam // 2
        # Weights
        weights = np.log(np.arange(lam, 0, -1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights ** 2)
        # Update learning rates with mu_eff
        cmu = (2 * (mu_eff - 2) + 1) / ((d + 2) ** 2 + mu_eff)
        cc = (4 + mu_eff / d) / (d + 4 + mu_eff / 2)
        c1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        cs = (mu_eff + 1) / (d + mu_eff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + cs
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        # State
        ps = np.zeros(d)
        hsig = 0.0
        steps = 0

        # Generate initial population
        def generate_population(m, sigma, B, D):
            z = rng.standard_normal((lam, d))
            x = m[None, :] + sigma * (z @ (B * D)[None, :])
            return np.clip(x, -5, 5)

        # First evaluation
        pop = generate_population(m, sigma, B, D)
        vals = np.array([func(x) for x in pop])
        steps += lam
        idx = np.argsort(vals)
        x_best = pop[idx[0]]
        f_best = vals[idx[0]]

        while steps < budget:
            # Update mean
            m_new = m + np.sum(weights[:, None] * pop[idx[:mu]], axis=0)
            # Update sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / chi_n * (
                (m_new - m) / (sigma * np.sqrt(np.diag(B @ np.diag(D ** 2) @ B.T)))
            )
            hsig = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (steps / lam + 1))) < (
                1.4 + 2 / (d + 1)
            ) * chi_n else 0.0
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)

            # Update covariance
            v = (m_new - m) / sigma
            C = (1 + c1 * (2 - c1) + cmu * (hsig - 1) - c1 - cmu) * C
            C += c1 * (np.outer(v, v) - C)
            C += cmu * np.sum(
                weights[i] * np.outer(
                    (pop[idx[i]] - m) / sigma, (pop[idx[i]] - m) / sigma
                )
                for i in range(mu)
            )
            # Symmetrize
            C = (C + C.T) / 2

            # Recompute decomposition periodically
            if steps % (10 * lam) < lam:
                B, D = np.linalg.eigh(C)

            m = m_new
            # Generate new population
            remaining = budget - steps
            if remaining < lam:
                break
            pop = generate_population(m, sigma, B, D)
            vals = np.array([func(x) for x in pop])
            steps += lam
            idx = np.argsort(vals)
            if vals[idx[0]] < f_best:
                f_best = vals[idx[0]]
                x_best = pop[idx[0]]

        return x_best, f_best
# EVOLVE-BLOCK-END
