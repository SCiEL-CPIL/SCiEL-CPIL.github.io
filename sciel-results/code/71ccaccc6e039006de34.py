"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with adaptive step size for bent cigar function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.start_time = time.time()
        self.deadline = self.start_time + 290.0  # 300s limit, 10s safety margin
        self.calls = 0
        self.dim = self.dim

        # Initialize
        x = np.zeros(self.dim)
        fx = func(x)
        self.calls += 1
        best_x = x.copy()
        best_fx = fx

        # CMA-ES parameters
        lambda_ = 4 * self.dim  # population size
        mu = lambda_ // 2
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
        c_cov = (2 + 2 * mu_eff / (self.dim + 1)) / (self.dim + 1 + 2 * mu_eff / (self.dim + 1))

        sigma = 2.0
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        C = np.eye(self.dim)

        def sample_from_cov(C, sigma, rng):
            # Cholesky decomposition for sampling
            try:
                L = np.linalg.cholesky(C)
                z = rng.standard_normal(self.dim)
                return sigma * (L @ z)
            except np.linalg.LinAlgError:
                # Fallback to diagonal
                return sigma * rng.standard_normal(self.dim)

        rng = np.random.default_rng(42)

        while self.calls < self.budget:
            # Check deadline
            if time.time() > self.deadline:
                break

            # Generate population
            pop = []
            for i in range(lambda_):
                step = sample_from_cov(C, sigma, rng)
                y = x + step
                y = np.clip(y, -5, 5)
                fy = func(y)
                self.calls += 1
                pop.append((fy, y))

            # Sort by fitness
            pop.sort(key=lambda t: t[0])

            # Update best
            if pop[0][0] < best_fx:
                best_fx = pop[0][0]
                best_x = pop[0][1].copy()

            # Compute weighted mean
            new_x = np.zeros(self.dim)
            for i in range(mu):
                new_x += weights[i] * pop[i][1]
            
            # Update covariance matrix adaptation
            diff = new_x - x
            x = new_x

            # Update path for sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C @ (diff / sigma))
            norm_p_sigma = np.linalg.norm(p_sigma)
            chi_n = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (24 * self.dim ** 2))
            
            # Update sigma
            sigma *= np.exp((c_sigma / d_sigma) * (norm_p_sigma / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e6)

            # Update path for covariance
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (diff / sigma)

            # Rank-one and rank-mu updates for C
            C = (1 + c_cov * (mu_eff + 1) - c_cov) * C
            C += c_cov * np.outer(p_c, p_c)
            C += c_cov * (1 - (1 - c_cov) ** (-1) - c_cov) * np.eye(self.dim)
            
            # Rank-mu update
            for i in range(mu):
                C += c_cov * weights[i] * np.outer(pop[i][1] - x, pop[i][1] - x)

            # Symmetrize
            C = (C + C.T) / 2

        return best_x, best_fx
# EVOLVE-BLOCK-END
