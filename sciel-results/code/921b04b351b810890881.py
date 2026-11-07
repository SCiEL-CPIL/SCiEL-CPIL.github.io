"""Two-stage optimizer: Sobol grid search + CMA-ES local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Two-stage: Sobol-based global search then CMA-ES local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def _sobol_sequence(self, n, dim, rng):
        """Generate a quasi-random Sobol-like sequence using a simple LCG-based
        Halton-like approach for reproducibility and speed.
        We use a prime-based Halton sequence which is deterministic and fast."""
        points = np.zeros((n, dim))
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        for i in range(n):
            for d in range(dim):
                p = primes[d]
                f, r = 1.0, i + 1
                while r > 0:
                    r, rem = divmod(r, p)
                    f /= p
                    if rem > 0:
                        f += rem * f
                points[i, d] = f
        return points

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin

        dim = self.dim
        budget = self.budget

        # Stage 1: Global search with stratified/Halton points
        n_global = min(100, budget // 5)
        n_local = budget - n_global

        points = self._sobol_sequence(n_global, dim, None)
        # Map [0,1]^dim to [-5,5]^dim
        points = points * 10.0 - 5.0

        best_x = None
        best_f = np.inf
        all_points = []
        all_values = []

        for i in range(n_global):
            if time.time() > deadline:
                break
            x = points[i]
            fx = func(x)
            all_points.append(x)
            all_values.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Stage 2: CMA-ES local refinement
        # Simple CMA-ES implementation
        x_mean = best_x.copy()
        sigma = 1.0
        # Initialize covariance matrix as identity
        C = np.eye(dim)
        # Inverse square root of C (identity initially)
        C_inv_sqrt = np.eye(dim)

        # Population size
        lam = max(4, int(4 + 3 * np.log(dim)))
        mu = lam // 2
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        # Step size adaptation
        cs = (mueff + 1) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, (mueff - 1) / (dim + 1)) - cs
        # Covariance matrix adaptation
        cmu = min(1, (mueff - 1 + np.sum(weights[:min(mu, 2)])) / (dim + 2) - cs)
        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        for gen in range(n_local):
            if time.time() > deadline:
                break

            # Generate population
            offspring = np.zeros((lam, dim))
            for i in range(lam):
                # Sample from N(x_mean, sigma^2 * C)
                # Use Cholesky-like: x_mean + sigma * C_inv_sqrt.T @ z
                # Actually we need C^{1/2}. Let's use eigen decomposition periodically.
                # For simplicity and speed, use a simplified approach:
                # Sample z ~ N(0, I), then y = x_mean + sigma * (C^{1/2} @ z)
                # We'll update C^{1/2} via eigen decomposition each generation (dim=5, cheap)
                evals = np.linalg.eigh(C)
                eigvals = np.clip(evals[0], 1e-12, None)
                C_half = evals[1] @ np.diag(np.sqrt(eigvals)) @ evals[1].T
                z = np.random.default_rng(hash((gen, i)) & 0xFFFFFFFF).standard_normal(dim)
                y = x_mean + sigma * (C_half @ z)
                y = np.clip(y, -5, 5)
                offspring[i] = y

            # Evaluate
            fitness = np.array([func(y) for y in offspring])

            # Sort
            idx = np.argsort(fitness)
            fitness_sorted = fitness[idx]
            x_sorted = offspring[idx]

            # Update best
            if fitness_sorted[0] < best_f:
                best_f = fitness_sorted[0]
                best_x = x_sorted[0].copy()

            # Update mean
            x_new = np.sum(weights[:, None] * x_sorted[:mu], axis=0)
            # CMA-ES update
            y_vec = (x_new - x_mean) / sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * y_vec
            norm_ps = np.linalg.norm(p_sigma)
            if norm_ps > (1.4 + 2 / (dim + 1)) * chi_n:
                pass  # Don't adapt sigma if path too long

            h_sigma = 1.0 if norm_ps < (1.4 + 2 / (dim + 1)) * chi_n * np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) else 0.0
            p_c = (1 - cs) * p_c + h_sigma * np.sqrt(cs * (2 - cs) * mueff) * y_vec
            sigma *= np.exp(cs / ds * (norm_ps / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update covariance
            delta_h = (2 - h_sigma) * cs * (1 - cs)
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (x_sorted[i] - x_mean) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)
            C = (1 + cs * ds * (1 - cs) - cmu) * C + cmu * (rank_mu + delta_h * rank_one)

            # Symmetrize and ensure positive definite
            C = (C + C.T) / 2
            # Regularize
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            x_mean = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
