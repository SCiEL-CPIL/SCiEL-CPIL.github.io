"""CMA-ES optimizer for BBOB-style bent cigar function."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time guard for the bent cigar (f12) problem."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        # lambda: population size
        lam = 4 + int(3 * np.log(dim))  # standard CMA-ES lambda
        mu = lam // 2  # number of recombination weights
        # recombination weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()

        # step size adaptation (CSA)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma

        # covariance matrix adaptation
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 / ((dim + 1.3) ** 2 + mu_eff)) * (2 - c_c + min(1, c_sigma))
        c_mu = min(1 - c_1, (2 * (mu_eff - 2 + 1 / mu_eff)) / ((dim + 2) ** 2 + mu_eff))

        # chi_lambda: expected norm of N(0, I) for lambda samples
        chi_lambda = np.sqrt(lam) * (1 - 1 / (4 * lam) + 1 / (21 * lam ** 2))

        # initial population
        x_best = rng.uniform(-4, 4, dim)
        f_best = func(x_best)
        x_mean = x_best.copy()
        sigma = 2.0

        # evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        # Precompute Cholesky of C
        try:
            C_sqrt = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C_sqrt = np.eye(dim)

        # Track best
        best_x = x_best.copy()
        best_f = f_best
        calls_made = 1

        # Counter for stagnation
        no_improve_count = 0
        stagnation_threshold = 50

        for gen in range(1, budget):
            # Check time
            if time.time() > deadline:
                break

            # Check if we've used enough budget
            if calls_made >= budget:
                break

            # Generate population
            offspring = []
            for i in range(lam):
                # Sample from N(x_mean, sigma^2 * C)
                z = rng.standard_normal(dim)
                # Use Cholesky decomposition for correlated sampling
                try:
                    d = C_sqrt @ z
                except Exception:
                    d = z
                y = np.clip(x_mean + sigma * d, -5, 5)
                fy = func(y)
                calls_made += 1
                offspring.append((fy, y))

            # Check time again
            if time.time() > deadline:
                break

            # Sort offspring by fitness
            offspring.sort(key=lambda t: t[0])

            # Update best
            if offspring[0][0] < best_f:
                best_f = offspring[0][0]
                best_x = offspring[0][1].copy()
                no_improve_count = 0
            else:
                no_improve_count += 1

            # Recombinant mean
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * offspring[i][1]

            # Update sigma (CSA)
            # First, update evolution path p_sigma
            # Need C^{-1/2}
            try:
                C_inv_sqrt = np.linalg.inv(C_sqrt)
            except Exception:
                C_inv_sqrt = np.eye(dim)

            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (y_w - x_mean) / sigma
            p_sigma_norm = np.linalg.norm(p_sigma)

            sigma *= np.exp(c_sigma / d_sigma * (p_sigma_norm / chi_lambda - 1))
            sigma = max(sigma, 1e-12)
            sigma = min(sigma, 100.0)

            # Update C (only if p_sigma is not too large)
            if p_sigma_norm / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_lambda:
                # Update evolution path p_c
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_w - x_mean) / sigma

                # Update covariance matrix
                delta_h = (1 + h_sig) * c_c * (2 - c_c)  # h_sig is indicator
                # Simplified: just use the rank-1 + rank-mu update
                C = (1 + c_c * (2 - c_c) - c_1 - c_mu) * C
                C += c_c * (np.outer(p_c, p_c) - C)
                C += c_1 * (np.outer((y_w - x_mean) / sigma, (y_w - x_mean) / sigma))

                # Rank-mu update
                for i in range(mu):
                    y_diff = (offspring[i][1] - x_mean) / sigma
                    C += c_mu * weights[i] * np.outer(y_diff, y_diff)

                # Ensure symmetry and positive semi-definite
                C = (C + C.T) / 2
                # Add small regularization for numerical stability
                C += 1e-12 * np.eye(dim)

                # Update Cholesky
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback: use eigenvalue decomposition
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))

            # Update mean
            x_mean = y_w

            # If stagnating, try a restart with larger sigma
            if no_improve_count > stagnation_threshold:
                # Perturb the mean
                x_mean = np.clip(x_mean + 0.5 * rng.standard_normal(dim), -5, 5)
                sigma = max(sigma * 2.0, 1.0)
                C = np.eye(dim)
                C_sqrt = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                no_improve_count = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
