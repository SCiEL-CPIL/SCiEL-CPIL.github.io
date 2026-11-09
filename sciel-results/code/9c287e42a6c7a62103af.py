"""CMA-ES optimizer for BBOB f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive covariance matrix for sharp ridge functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        N = max(4, dim + int(round(3 * np.log(dim))))  # population size
        lambda_ = N
        mu = lambda_ // 2  # number of selected individuals

        # Weights (logarithmic)
        w = np.log(np.arange(mu, 0, -1)) - np.log(np.arange(mu, 0, -1) - 1)
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        # Step size control
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 - mu_eff / (dim + 1) ** 2 + 2 * mu_eff / dim) / (
            (dim + 2) ** 2 + mu_eff / dim
        )
        c_mu = min(
            1 - c_1,
            (2 * (mu_eff - 2 + 1 / mu_eff)) / ((dim + 2) ** 2 + mu_eff),
        )

        # Initial parameters
        x0 = np.zeros(dim)
        sigma0 = 2.0

        # Initial state
        x_mean = x0.copy()
        sigma = sigma0
        C = np.eye(dim)  # covariance matrix
        C_inv_sqrt = np.eye(dim)  # C^{-1/2}
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Track best
        best_x = x0.copy()
        best_f = float('inf')

        # Evaluate initial point
        f_calls = 0

        def evaluate(x):
            nonlocal f_calls, best_x, best_f
            if f_calls >= budget:
                return best_f
            f_calls += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # First evaluation
        fx = evaluate(x_mean)
        if not np.isfinite(fx):
            fx = 1e10

        # Precompute eigenvalues for C^{-1/2} updates
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.clip(eigvals, 1e-20, None)

        for iteration in range(budget - 1):
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            Y = np.array([rng.standard_normal(dim) for _ in range(lambda_)])
            # Transform using C^{-1/2}
            Z = Y @ C_inv_sqrt.T  # each row is C^{-1/2} @ y_i
            X = np.array([x_mean + sigma * z for z in Z])

            # Clip to bounds
            X = np.clip(X, -5, 5)

            # Evaluate population
            F = np.array([evaluate(x) for x in X])

            # Sort by fitness
            indices = np.argsort(F)
            X_sorted = X[indices]
            F_sorted = F[indices]

            # Update mean (weighted recombination)
            x_mean_new = np.sum(w[:, None] * X_sorted[:mu], axis=0)

            # Update step size (CSA)
            y_sigma = (x_mean_new - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            h_sigma = (
                np.linalg.norm(p_sigma)
                / np.sqrt(1 - (1 - c_sigma) ** (2 * (iteration + 1)))
                / chi_n
            )
            h_sigma = 1.0 if h_sigma < 1.4 + 2 / (dim + 1) else 0.0
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(sigma, 1e-12)

            # Update evolution path
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma

            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.sum(w[:, None, None] * np.outer(X_sorted[:mu] - x_mean, X_sorted[:mu] - x_mean) / sigma**2, axis=0)
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, 1e12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            x_mean = x_mean_new

            # Safety: if sigma becomes too small or too large
            if sigma < 1e-10 or sigma > 100:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
