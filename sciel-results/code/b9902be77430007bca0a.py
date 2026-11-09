"""Rewritten optimizer: CMA-ES for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for minimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s kill

        # CMA-ES parameters
        popsize = 4 + int(3 * np.log(dim))  # standard CMA-ES popsize
        popsize = max(popsize, 5)

        # Initialize at origin (center of box)
        x = np.zeros(dim)
        f_x = func(x)
        best_x = x.copy()
        best_f = f_x
        calls = 1

        sigma = 4.0  # large initial step to explore the box
        C = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Update rates
        mu = popsize // 2  # number of recombination weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        def clip_box(v):
            return np.clip(v, -5.0, 5.0)

        for gen in range(budget - 1):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            # Update C and sigma (CMA-ES update)
            # Compute covariance update
            rank_mu = np.zeros((dim, dim))
            # We'll compute the offspring first, then update

            # Sample offspring
            B = np.linalg.cholesky(C)
            offspring = []
            for i in range(popsize):
                z = np.random.standard_normal(dim)
                y = x + sigma * B @ z
                y = clip_box(y)
                f_y = func(y)
                calls += 1
                if f_y < best_f:
                    best_f = f_y
                    best_x = y.copy()
                offspring.append((y, f_y))
                if calls >= budget:
                    break

            if len(offspring) < 2:
                break

            # Sort offspring by fitness
            offspring.sort(key=lambda t: t[1])
            selected = offspring[:mu]

            # Weighted mean of selected
            m_new = np.zeros(dim)
            for i, (y, f_y) in enumerate(selected):
                m_new += weights[i] * y
            m_new /= np.sum(weights[:mu])

            # Update evolution paths
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ (m_new - x) / sigma)
            norm_ps = np.linalg.norm(ps)
            h_sigma = 1.0 if norm_ps / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - gen * popsize))) < (1 + 2 / (dim + 1)) * chi_n else 0.0
            pc = (1 - c_c) * pc + h_sigma * c_c * (m_new - x) / sigma

            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_sigma * (c_sigma + d_sigma - 1) - delta_h) * C
            for i, (y, f_y) in enumerate(selected):
                diff = (m_new - x) / sigma - B.T @ ((y - x) / sigma)
                C += c_sigma * weights[i] * (np.outer(diff, diff) + np.outer(y - x, y - x) / sigma ** 2 - C)

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update mean
            x = m_new

            # Ensure C is symmetric and positive semi-definite
            C = (C + C.T) / 2
            # Check eigenvalues
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-12:
                C += (1e-12 - np.min(eigvals)) * np.eye(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
