"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with adaptive step size and population sampling."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # safety margin before 300s limit

        # Population size
        lam = max(4, int(np.ceil(4 + 3 * np.log(dim))))
        # For dim=5, lam = max(4, ceil(4 + 3*ln(5))) = max(4, ceil(4+4.83)) = max(4, 9) = 9
        mu = lam // 2

        # Initial mean and sigma
        x = rng.uniform(-4, 4, dim)
        sigma = 2.0
        fx = func(x)

        # Track best
        best_x = x.copy()
        best_fx = fx

        # Covariance matrix (identity initially)
        C = np.eye(dim)
        # Step size adaptation (1/5 rule variant)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        c_sigma = (mu + 0.5) / (dim + mu + 2.5)
        d_sigma = 1 + 2 * max(0, (sqrt_dim + mu/2 - 1) / (dim + sqrt_dim + 2)) if False else 1
        sqrt_dim = np.sqrt(dim)
        d_sigma = 1 + 2 * max(0, (sqrt_dim + mu/2 - 1) / (dim + sqrt_dim + 2))
        c_c = (4 + dim/5) / (dim + 4 + 2*mu/dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2/mu) / ((dim + 2)**2 + mu_eff))
        mu_eff = 2.0 / np.sum(np.arange(1, mu+1)**-2)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2/mu) / ((dim + 2)**2 + mu_eff))

        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu+1))
        weights /= np.sum(weights)

        evals_used = 1

        while evals_used < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            Y = np.zeros((lam, dim))
            for i in range(lam):
                Y[i] = x + sigma * (C @ rng.standard_normal(dim))
                Y[i] = np.clip(Y[i], -5, 5)

            # Evaluate
            F = np.array([func(Y[i]) for i in range(lam)])
            evals_used += lam

            # Update best
            idx_min = np.argmin(F)
            if F[idx_min] < best_fx:
                best_fx = F[idx_min]
                best_x = Y[idx_min].copy()

            # Sort
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]

            # Weighted mean of best mu
            y_mean = np.zeros(dim)
            for i in range(mu):
                y_mean += weights[i] * (Y_sorted[i] - x) / sigma

            # Update p_sigma
            h_sig = 1.0 if (np.dot(y_mean, y_mean) / (1 - np.exp(-2 * evals_used / (dim + c_sigma))) < (1 + 2/mu) * chi_n**2) else 0.0
            p_sigma = (1 - c_sigma) * p_sigma + h_sig * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean

            # Update p_c
            h_c = 1.0 if (np.dot(p_sigma, p_sigma) / (1 - np.exp(-2 * evals_used / c_c)) < (1 + 2/mu) * chi_n**2) else 0.0
            p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean

            # Update covariance
            outer = np.zeros((dim, dim))
            for i in range(mu):
                diff = (Y_sorted[i] - x) / sigma
                outer += weights[i] * np.outer(diff, diff)
            C = (1 + c_1 * h_c * (2 - c_c) * (1 - np.dot(p_c, p_c) / chi_n**2) + c_mu * h_c * (2 - c_c) * (1 - np.dot(p_c, p_c) / chi_n**2) - c_1 - c_mu) * C + \
                c_1 * (np.outer(p_c, p_c) + (1 - h_c) * (1 - np.exp(-2 * evals_used / c_c)) * np.eye(dim)) + \
                c_mu * outer

            # Symmetrize and ensure positive definite
            C = (C + C.T) / 2
            # Add small regularization if needed
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-12:
                C += 1e-12 * np.eye(dim)

            # Update mean
            x = x + sigma * y_mean
            x = np.clip(x, -5, 5)

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)

        # Final polish: use remaining budget with smaller steps around best
        if evals_used < budget:
            remaining = budget - evals_used
            x_polish = best_x.copy()
            fx_polish = best_fx
            sigma_polish = 0.1
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                y = np.clip(x_polish + sigma_polish * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                if fy < fx_polish:
                    x_polish, fx_polish = y, fy
                    sigma_polish *= 1.2
                else:
                    sigma_polish *= 0.9
                if fx_polish < best_fx:
                    best_fx = fx_polish
                    best_x = x_polish.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
