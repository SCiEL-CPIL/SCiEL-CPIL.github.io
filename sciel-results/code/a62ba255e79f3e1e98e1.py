"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with PCA-seeded covariance for ill-conditioned landscapes."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit

        # CMA-ES parameters tuned for dim=5, budget=500
        # lambda = 4 is a good balance: enough exploration without wasting budget
        lam = 4
        mu = lam // 2  # top-2 weights
        # Standard weights: w_i = ln(mu+0.5) - ln(i) for i=1..mu, normalized
        weights = np.array([np.log(mu + 0.5) - np.log(i + 1) for i in range(mu)])
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)

        # CMA-ES step size control parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))

        # Initial population: spread over the box to capture the funnel shape
        rng = np.random.default_rng(42)
        pop_size = 16  # initial exploration population
        X0 = rng.uniform(-4, 4, (pop_size, dim))
        F0 = np.array([func(x) for x in X0])

        # Find best point from initial exploration
        idx_best = np.argmin(F0)
        x_mean = X0[idx_best].copy()
        f_best = F0[idx_best]
        calls_used = pop_size

        # PCA covariance from early samples to align with anisotropic landscape
        # Use the top half of samples for better covariance estimate
        n_pca = max(4, pop_size // 2)
        sorted_idx = np.argsort(F0)
        good_samples = X0[sorted_idx[:n_pca]]
        # Center and compute covariance
        centered = good_samples - good_samples.mean(axis=0)
        if n_pca > 1:
            cov = (centered.T @ centered) / (n_pca - 1)
            # Regularize
            cov += 1e-4 * np.eye(dim)
        else:
            cov = np.eye(dim)

        # Cholesky decomposition for sampling
        try:
            C_chol = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            C_chol = np.eye(dim)

        sigma = 1.0  # initial step size
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Track best
        best_x = x_mean.copy()
        best_f = f_best

        # Main CMA-ES loop
        while calls_used < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate lambda offspring
            # Sample from N(x_mean, sigma^2 * C)
            Z = rng.standard_normal((lam, dim))
            Y = x_mean + sigma * (Z @ C_chol.T)
            # Clip to box
            Y = np.clip(Y, -5, 5)

            # Evaluate
            FY = np.array([func(y) for y in Y])
            calls_used += lam

            # Sort by fitness
            sorted_idx = np.argsort(FY)
            Y_sorted = Y[sorted_idx]
            FY_sorted = FY[sorted_idx]

            # Update best
            if FY_sorted[0] < best_f:
                best_f = FY_sorted[0]
                best_x = Y_sorted[0].copy()

            # Weighted mean of top mu
            x_new = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)

            # Update covariance matrix
            y_vec = (x_new - x_mean) / sigma
            h_sigma = (np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_used + 1) / (dim + c_sigma))) - (1 - c_sigma)) / (c_sigma * np.sqrt(mu_eff))
            norm_y = np.linalg.norm(y_vec)
            if norm_y > 1e-10:
                p_sigma = (1 - c_sigma) * p_sigma + h_sigma * c_sigma * norm_y * y_vec / norm_y
            else:
                p_sigma = (1 - c_sigma) * p_sigma

            # Update evolution path for covariance
            norm_pc = np.linalg.norm(p_c)
            c_c_sqrt = np.sqrt(c_c * (4 - c_c) * mu_eff)
            if norm_pc > 1e-10:
                p_c = (1 - c_c) * p_c + c_c_sqrt * (1 + (1 - c_sigma) * h_sigma) * y_vec
            else:
                p_c = (1 - c_c) * p_c

            # Rank-one and rank-mu update
            v = (p_c + c_c_sqrt * (1 - c_c) * y_vec) / (1 + (1 - c_c) * h_sigma)
            C = (1 + c_1 * (2 - c_1) + c_mu) * C \
                - c_1 * np.outer(p_c, p_c) \
                - c_mu * np.outer(v, v) \
                + (c_1 + c_mu) * np.eye(dim)

            # Rank-mu update
            for i in range(mu):
                diff = Y_sorted[i] - x_mean
                C += c_mu * weights[i] * np.outer(diff, diff)

            # Re-normalize to maintain trace
            tr = np.trace(C)
            if tr > 0:
                C /= tr

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))

            # Update x_mean
            x_mean = x_new

            # Recompute Cholesky periodically to maintain stability
            if calls_used % lam == 0:
                try:
                    C_chol = np.linalg.cholesky(C + 1e-6 * np.eye(dim))
                except np.linalg.LinAlgError:
                    C_chol = np.eye(dim)
                    C = np.eye(dim)

            # Stagnation check: if no improvement in last 50 calls, reduce sigma
            # and possibly restart with smaller sigma
            if calls_used % 50 == 0:
                if sigma < 0.01:
                    # Very small sigma, do a local refinement
                    sigma = 0.5
                    # Reset paths
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
