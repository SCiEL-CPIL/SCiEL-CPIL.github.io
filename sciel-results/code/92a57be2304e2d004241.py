"""Rewritten optimizer for BBOB f8 (Rosenbrock) with adaptive population-based ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Adaptive CMA-ES variant for 5D Rosenbrock with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim

        # CMA-ES parameters
        lambda_pop = max(4, int(4 + 3 * np.log(dim)))  # population size
        c1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 1) ** 2 + 2)
        cmu = min(0.04, c1)
        cs = (dim + 1.3) / (dim + 5.5)
        ds = 1 + 2 * max(0, np.log((cs - 0.5) / (cs + 1.5) + 1))
        chiN = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # Initialize
        x0 = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        def sqrtm_inverse_sqrt(C_mat):
            """Compute B * D^{-1/2} where C = B diag(D^2) B^T"""
            eigvals, eigvecs = np.linalg.eigh(C_mat)
            eigvals = np.clip(eigvals, 1e-12, None)
            D_inv_sqrt = 1.0 / np.sqrt(eigvals)
            return eigvecs * D_inv_sqrt  # B * diag(D^{-1/2})

        def sample(sigma_val, B_mat, D_vec):
            """Sample from N(0, sigma^2 * C)"""
            z = rng.standard_normal(dim)
            return sigma_val * (B_mat @ (D_vec * z))

        # Initial population
        pop = [x0 + sample(sigma, B, D) for _ in range(lambda_pop)]
        pop = [np.clip(p, -5, 5) for p in pop]
        fvals = np.array([func(p) for p in pop])
        best_idx = np.argmin(fvals)
        x_best = pop[best_idx].copy()
        f_best = fvals[best_idx]
        calls = lambda_pop

        # Sort by fitness
        order = np.argsort(fvals)
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        for iteration in range(self.budget):
            if time.time() > deadline:
                break

            # Update mean
            m_new = sum(weights[i] * pop[order[i]] for i in range(mu))

            # Update step size control
            y = (m_new - x0) / (sigma * chiN)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (iteration + 1)) / chiN) < (1.4 + 2 / (dim + 1))

            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chiN - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update covariance
            pc = (1 - c1) * pc + hsig * np.sqrt(c1 * (2 - c1) * mueff) * (m_new - x0) / sigma
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * c1 * C)
            for i in range(mu):
                yi = (pop[order[i]] - x0) / sigma
                C += cmu * weights[i] * np.outer(yi, yi)

            # Ensure symmetry and positive definiteness
            C = 0.5 * (C + C.T)
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update eigen decomposition
            B, D = eigvecs, np.sqrt(eigvals)

            # Update mean
            x0 = m_new

            # Generate new population
            pop = [x0 + sample(sigma, B, D) for _ in range(lambda_pop)]
            pop = [np.clip(p, -5, 5) for p in pop]
            fvals = np.array([func(p) for p in pop])
            calls += lambda_pop

            # Track best
            best_idx = np.argmin(fvals)
            if fvals[best_idx] < f_best:
                f_best = fvals[best_idx]
                x_best = pop[best_idx].copy()

            # If budget exhausted, break
            if calls >= self.budget:
                break

        return x_best, f_best
# EVOLVE-BLOCK-END
