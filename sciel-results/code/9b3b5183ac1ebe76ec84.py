"""Rewritten optimizer using CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for the Bent cigar (f12) BBOB task."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Parameters
        # Population size: lambda = 4 + floor(3 * log(dim + 1))
        lam = int(4 + 3 * np.log(dim + 1))
        # Weights: log-linear weights
        mu = lam // 2  # number of best individuals
        w = np.log(np.arange(mu, 0, -1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialize
        x_mean = np.zeros(dim)
        # Start with a reasonable sigma; for bent cigar, larger sigma helps initially
        sigma = 2.0
        B = np.eye(dim)
        D = 1.0
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        best_x = None
        best_f = np.inf

        # Evaluate initial mean
        x_mean_clipped = np.clip(x_mean, -5, 5)
        f_mean = func(x_mean_clipped)
        best_x = x_mean_clipped.copy()
        best_f = f_mean

        calls_made = 1
        remaining = budget - calls_made

        while remaining > 0:
            # Update covariance matrix adaptation
            # Compute C = B * diag(D^2) * B^T
            C = (B * D) @ B.T

            # Step size control
            x_old = x_mean.copy()

            # Generate population
            Y = []
            X = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                # Transform: B * D * z
                y = B @ (D * z)
                x_new = x_mean + sigma * y
                x_new = np.clip(x_new, -5, 5)
                Y.append(z)
                X.append(x_new)

            # Evaluate population
            Fs = []
            for x_new in X:
                if remaining <= 0:
                    break
                f_val = func(x_new)
                Fs.append(f_val)
                remaining -= 1

            if not Fs:
                break

            # Sort by fitness
            order = np.argsort(Fs)
            X_sorted = [X[i] for i in order]
            Fs_sorted = [Fs[i] for i in order]
            Y_sorted = [Y[i] for i in order]

            # Update best
            if Fs_sorted[0] < best_f:
                best_f = Fs_sorted[0]
                best_x = X_sorted[0].copy()

            # Compute weighted mean of best mu
            x_new_mean = sum(w[i] * X_sorted[i] for i in range(mu))

            # Update path for sigma control
            inv_sqrt_C = B @ (1.0 / D) @ B.T
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_sqrt_C @ (x_new_mean - x_old) / sigma)

            # Update path for covariance control
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (inv_sqrt_C @ (x_new_mean - x_old) / sigma)

            # Update covariance matrix
            C = (1 + c_1 * (c_mu + c_1) - c_1) * C \
                + c_1 * (p_c @ p_c + (c_mu + c_1 - c_1) * C) \
                + c_mu * sum(w[i] * np.outer(X_sorted[i] - x_old, X_sorted[i] - x_old) / (sigma ** 2) for i in range(mu))

            # Ensure symmetry
            C = (C + C.T) / 2

            # Eigen decomposition for stability
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                D = np.sqrt(eigvals)
                B = eigvecs
            except np.linalg.LinAlgError:
                # Fallback to identity if decomposition fails
                B = np.eye(dim)
                D = np.ones(dim)

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Clamp sigma to avoid extreme values
            sigma = np.clip(sigma, 1e-8, 50.0)

            # Update mean
            x_mean = x_new_mean

            # If budget is low, do a local refinement
            if remaining < lam:
                # Use remaining budget for fine-tuning around best_x
                local_sigma = max(sigma * 0.1, 1e-4)
                for _ in range(remaining):
                    x_try = best_x + local_sigma * rng.standard_normal(dim)
                    x_try = np.clip(x_try, -5, 5)
                    f_try = func(x_try)
                    remaining -= 1
                    if f_try < best_f:
                        best_f = f_try
                        best_x = x_try.copy()
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
