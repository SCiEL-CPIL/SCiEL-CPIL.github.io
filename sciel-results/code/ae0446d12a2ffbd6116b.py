"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES implementation for continuous optimization."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin before 300s limit

        rng = np.random.default_rng(0)
        dim = self.dim

        # CMA-ES parameters
        # lambda (population size)
        lam = int(4 + 3 * dim)
        # step size
        sigma = 1.0
        # initial mean
        m = rng.uniform(-5, 5, dim)
        fm = func(m)

        # covariance matrix
        C = np.eye(dim)
        # evolution path for covariance
        pc = np.zeros(dim)
        # evolution path for step size
        ps = np.zeros(dim)

        # adaptation rates
        mu = lam // 2  # number of recombination weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        # eigen decomposition of C for sampling
        def update_eigen(C):
            eigvals, eigvecs = np.linalg.eigh(C)
            # clip small negative values
            eigvals = np.clip(eigvals, 1e-12, None)
            return eigvals, eigvecs

        eigvals, eigvecs = update_eigen(C)

        best_x = m.copy()
        best_f = fm

        calls_made = 1

        while calls_made < self.budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Sample population
            pops = []
            popfs = []

            for i in range(lam):
                # Sample direction
                z = rng.standard_normal(dim)
                # Transform to correlated direction using eigen decomposition
                # y = eigvecs @ (sqrt(eigvals) * z)
                y = (eigvecs * np.sqrt(eigvals)) @ z
                x_i = m + sigma * y
                x_i = np.clip(x_i, -5, 5)
                pops.append(x_i)
                f_i = func(x_i)
                popfs.append(f_i)
                calls_made += 1

                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()

            # Sort by fitness
            order = np.argsort(popfs)
            sorted_pops = [pops[i] for i in order]
            sorted_fs = [popfs[i] for i in order]

            # Update mean
            m_new = np.sum([weights[i] * sorted_pops[i] for i in range(mu)], axis=0)
            m_new = np.clip(m_new, -5, 5)

            # Update evolution path for step size
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * \
                 (eigvecs.T @ (m_new - m) / (sigma * np.sqrt(np.sum(eigvals))))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (24 * dim ** 2))
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (calls_made + 1)) < (1.4 + 2 / (dim + 1)) * chi_n:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Update evolution path for covariance
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * (m_new - m) / sigma

            # Update covariance matrix
            delta_sig = (1 - c_sigma) ** (calls_made + 1)
            rank_one = np.outer(pc, pc)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                diff = (sorted_pops[i] - m) / sigma
                rank_mu += weights[i] * np.outer(diff, diff)

            C = (1 + c_1 * (delta_sig - 1) + c_mu) * C - c_1 * rank_one + c_mu * rank_mu

            # Ensure C is symmetric
            C = (C + C.T) / 2

            # Eigen decomposition
            eigvals, eigvecs = update_eigen(C)

            m = m_new

        return best_x, best_f
# EVOLVE-BLOCK-END
