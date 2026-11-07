"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for BBOB f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        # Population size
        lam = 4 + int(np.floor(3 * np.log(dim)))
        mu = lam // 2
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        # Step size control
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        # Covariance matrix adaptation
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        # Chi-squared approximation
        chi_lam = np.sqrt(lam) * (1 - 1 / (4 * lam) + 1 / (25 * lam ** 2))

        # Initial population
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        # Use eigendecomposition for C^{-1/2}
        # Initially C=I, so C^{-1/2}=I

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        best_x = x_mean.copy()
        best_f = np.inf
        calls_made = 0

        def sample_from_cma(mean, sigma, C_sqrt_inv):
            # Generate lambda samples
            z = rng.standard_normal((lam, dim))
            # Transform to have covariance C
            # x = mean + sigma * C^{1/2} * z
            # We store C^{-1/2} for updating, but for sampling we need C^{1/2}
            # Let's maintain C and use Cholesky or eigen for sampling
            return z

        # We'll use a simpler approach: maintain C and use eigendecomposition periodically
        # Or use the standard CMA-ES update with C^{1/2} via eigen

        # Let's implement standard CMA-ES
        # C is the covariance matrix
        # For sampling: x = mean + sigma * A @ z where A @ A.T = C
        # We'll use Cholesky decomposition for sampling

        for iteration in range(budget // lam + 1):
            if calls_made >= budget:
                break

            # Sample lambda points
            # Get Cholesky factor of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback to eigen if Cholesky fails
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                L = eigvecs @ np.diag(np.sqrt(eigvals))

            z = rng.standard_normal((lam, dim))
            # x = x_mean + sigma * L @ z.T
            offspring = (sigma * (z @ L.T) + x_mean).T
            offspring = np.clip(offspring, -5, 5)

            # Evaluate offspring
            f_values = np.zeros(lam)
            for i in range(lam):
                if calls_made >= budget:
                    break
                f_values[i] = func(offspring[i])
                calls_made += 1
                if f_values[i] < best_f:
                    best_f = f_values[i]
                    best_x = offspring[i].copy()

            # If we broke early due to budget, fill remaining with inf
            for i in range(calls_made - iteration * lam, lam):
                f_values[i] = np.inf

            # Sort by fitness
            idx = np.argsort(f_values)
            offspring = offspring[idx]
            f_values = f_values[idx]

            # Weighted mean of best mu
            new_mean = np.sum(weights[:, None] * offspring[:mu], axis=0)

            # Update sigma
            old_sigma = sigma
            # Compute h_sigma
            # C^{1/2} is L (Cholesky)
            # h_sigma = ||L^{-1} (new_mean - x_mean)|| / (c_sigma * chi_lam * sigma)
            # L^{-1} (new_mean - x_mean)
            diff = new_mean - x_mean
            # Solve L @ y = diff
            try:
                y = np.linalg.solve(L, diff)
            except np.linalg.LinAlgError:
                y = np.linalg.pinv(L) @ diff
            norm_y = np.linalg.norm(y)
            h_sigma = 1.0 if norm_y / (c_sigma * chi_lam * sigma) < (1.7 + 2 / (dim + 1)) * (1 - c_c) else 0.0

            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (L @ (diff / sigma))

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_lam - 1))
            # Limit sigma
            sigma = np.clip(sigma, 1e-12, 1e12)

            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * diff / sigma

            # Update C
            # Rank-mu update
            v = (new_mean - x_mean) / sigma
            C = ((1 + c_1 * c_mu - c_1) * C +
                 c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) +
                 c_mu * np.sum(weights[:, None] * np.outer(offspring[:mu] - x_mean, (offspring[:mu] - x_mean) / sigma), axis=0) / sigma)

            # Ensure symmetry and positive semi-definite
            C = (C + C.T) / 2
            # Regularize if needed
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            x_mean = new_mean
            x_mean = np.clip(x_mean, -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
