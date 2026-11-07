"""Rewritten optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time guard for 5-D box-constrained optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 300s limit, 20s safety margin

        dim = self.dim
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(n)))  # population size
        # Default step size
        sigma = 2.0

        # Initialize at center of box
        x_mean = np.zeros(n)
        # Initial covariance identity
        C = np.eye(n)
        # Precompute eigendecomposition for step generation
        D = np.ones(n)  # eigenvalues of C (all 1 initially)

        # CMA-ES control parameters
        mu = lambda_ // 2  # number of parents
        # Weights: linear weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        d_sigma = 1 + c_sigma + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # State variables
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        hs = chi_n * (1 + c_sigma)

        # Population
        pop = np.zeros((lambda_, n))
        fit = np.zeros(lambda_)

        # Evaluate initial point
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls_used = 1

        # Main loop
        while calls_used < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            # Use eigendecomposition of C for sampling
            eigvals, eigvecs = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(eigvals, 1e-12))
            
            for i in range(lambda_):
                z = np.random.randn(n)
                # Apply C^{1/2}: eigvecs @ (D * z) @ eigvecs.T
                x_i = x_mean + sigma * (eigvecs @ (D * z))
                # Clip to box
                x_i = np.clip(x_i, -5, 5)
                pop[i] = x_i
                fit[i] = func(x_i)
                calls_used += 1
                if time.time() > deadline:
                    # Fill remaining with current best to avoid issues
                    for j in range(i + 1, lambda_):
                        pop[j] = best_x
                        fit[j] = best_f
                    break

            # Check if we hit deadline mid-evaluation
            if time.time() > deadline:
                break

            # Sort by fitness
            idx = np.argsort(fit)
            pop_sorted = pop[idx]
            fit_sorted = fit[idx]

            # Update best
            if fit_sorted[0] < best_f:
                best_f = fit_sorted[0]
                best_x = pop_sorted[0].copy()

            # Weighted mean of selected parents
            x_new = np.zeros(n)
            for i in range(mu):
                x_new += weights[i] * pop_sorted[i]
            x_new /= weights.sum()

            # Update evolution paths
            # p_sigma
            p_sigma = (1 - c_sigma) * p_sigma
            # Need C^{-1/2} (x_new - x_mean) / sigma
            # C^{-1/2} = eigvecs @ diag(1/D) @ eigvecs.T
            C_inv_sqrt = eigvecs @ np.diag(1.0 / D) @ eigvecs.T
            diff = (x_new - x_mean) / sigma
            p_sigma += c_sigma * (C_inv_sqrt @ diff)

            # Check hsigma condition
            hs_new = (1 - c_sigma) * hs
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (calls_used // lambda_)) < \
               (1.4 + 2 / (n + 1)) * chi_n * hs_new:
                p_c = (1 - c_c) * p_c + c_c * (x_new - x_mean) / sigma
                hs = hs_new

            # Update mean
            x_mean = x_new

            # Update covariance
            delta_sigma = np.linalg.norm(p_sigma) / chi_n - 1
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                diff_i = (pop_sorted[i] - x_mean) / sigma
                rank_mu += weights[i] * np.outer(diff_i, diff_i)

            C = (1 + c_1 * (2 - c_1) - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            # Clamp sigma
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Ensure C is symmetric and positive semi-definite
            C = (C + C.T) / 2
            # Add small regularization
            C += 1e-12 * np.eye(n)

        return best_x, best_f
# EVOLVE-BLOCK-END
