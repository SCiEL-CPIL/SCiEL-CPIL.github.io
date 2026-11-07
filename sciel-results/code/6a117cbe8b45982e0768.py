"""Rewritten optimizer using CMA-ES style covariance adaptation."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES style continuous optimizer for BBOB f24."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget

        # Strategy parameters
        lambda_ = max(4, int(4 + 3 * np.log(n)))  # population size
        # CMA-ES step size control
        sigma = 2.0
        # Adaptation rates
        c_sigma = (lambda_ + 2) / (n + lambda_ + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((lambda_ / (n + 1) - 1)) / (n + 1) - 1) + c_sigma
        c_c = (4 + lambda_ / n) / (n + 4 + 2 * lambda_ / n)
        c_1 = 2 / ((n + 1.3) ** 2 + lambda_)
        c_mu = min(1 - c_1, 2 * ((lambda_ / 2 + 1) / (n + 2) - 1))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Initial point: center of the box, which is often good for BBOB
        x_mean = np.zeros(n)
        # Evaluate initial point
        f_vals = [func(x_mean)]
        calls = 1

        # Weighting for mean update
        weights = np.log(np.arange(lambda_, 0, -1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Covariance matrix
        C = np.eye(n)
        # Eigen decomposition cache
        eigvals, eigvecs = np.linalg.eigh(C)
        # Precompute C^(1/2)
        C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-10))) @ eigvecs.T
        C_sqrt = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 1e-10))) @ eigvecs.T

        # Cumulation paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        best_x = x_mean.copy()
        best_f = f_vals[0]

        for gen in range((budget - 1) // lambda_ + 1):
            if calls >= budget:
                break

            # Generate population
            Y = []
            X = []
            F = []
            for i in range(lambda_):
                if calls >= budget:
                    break
                z = np.random.default_rng(hash((gen, i)) & 0xFFFFFFFF).standard_normal(n)
                y = x_mean + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                f = func(y)
                calls += 1
                Y.append(z)
                X.append(y)
                F.append(f)

            if len(F) == 0:
                break

            # Sort by fitness
            order = np.argsort(F)
            X_sorted = [X[i] for i in order]
            F_sorted = [F[i] for i in order]
            Y_sorted = [Y[i] for i in order]

            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = X_sorted[0].copy()

            # Truncate to mu_ = lambda_ // 2 or all if small
            mu = min(len(F_sorted), max(1, lambda_ // 2))
            w = weights[:mu]
            w_sum = w.sum()
            w = w / w_sum
            mu_eff = 1.0 / np.sum(w ** 2)

            # Update mean
            x_new = np.zeros(n)
            for i in range(mu):
                x_new += w[i] * X_sorted[i]
            x_mean = x_new

            # Update sigma
            y_sigma = (C_sqrt_inv @ (x_new - x_mean)) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - gen * lambda_)))) / chi_n * (1 - c_sigma)
            if h_sigma < 1.4:
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(1e-8, min(sigma, 1e5))

            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (1 - c_c)
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_sigma
            C = ((1 + c_1 * delta_h - c_1 - c_mu) * C +
                 c_1 * np.outer(p_c, p_c) +
                 c_mu * sum(w[i] * np.outer(Y_sorted[i], Y_sorted[i]) for i in range(mu)))

            # Re-orthogonalize C to maintain positive semi-definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        return best_x, best_f
# EVOLVE-BLOCK-END
