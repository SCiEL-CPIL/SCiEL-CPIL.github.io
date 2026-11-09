"""Rewritten optimizer using CMA-ES with adaptive step size."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Parameters
        popsize = max(4, int(4 + 3 * np.log(n)))  # Standard CMA-ES population size
        c_sigma = (mu + 1) / ((n + 1.3) ** 2 + mu)  # Step size adaptation
        c_c = (4 + mu / n) / (n + 4 + 2 * mu / n)  # Path for correlation matrix
        c_1 = 2 / ((n + 1.3) ** 2 + mu)  # Rank-one update
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / (n + 1)) / ((n + 2) ** 2 + mu_eff))  # Rank-mu update
        mu = popsize // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Initial state
        x_mean = rng.uniform(-5, 5, n)
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        inv_sqrt_C = np.eye(n)

        # Evaluate initial point
        x_clipped = np.clip(x_mean, -5, 5)
        f_best = func(x_clipped)
        x_best = x_clipped.copy()

        calls_made = 1

        def evaluate(x):
            nonlocal calls_made, f_best, x_best
            x_c = np.clip(x, -5, 5)
            f = func(x_c)
            calls_made += 1
            if f < f_best:
                f_best = f
                x_best = x_c.copy()
            return f, x_c

        # Main loop
        while calls_made < budget:
            # Generate population
            Y = []
            for i in range(popsize):
                # Sample from N(0, C)
                z = rng.standard_normal(n)
                y = x_mean + sigma * (inv_sqrt_C @ z)
                Y.append(y)
            Y = np.array(Y)

            # Evaluate population
            F = []
            X_eval = []
            for y in Y:
                f_val, x_c = evaluate(y)
                F.append(f_val)
                X_eval.append(x_c)
                if calls_made >= budget:
                    break

            if calls_made >= budget:
                break

            F = np.array(F)
            X_eval = np.array(X_eval)

            # Sort by fitness
            order = np.argsort(F)
            X_sorted = X_eval[order]
            F_sorted = F[order]

            # Update mean
            x_new_mean = np.sum(weights[:, None] * X_sorted[:mu], axis=0)

            # Update path covariance
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * inv_sqrt_C @ (x_new_mean - x_mean) / sigma

            # Update step size (CSA - Cumulative Step-size Adaptation)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * inv_sqrt_C @ (x_new_mean - x_mean) / sigma
            hsig = min(1, (np.linalg.norm(ps) / (1 - (1 - c_sigma) ** (budget - calls_made + 1))) / (chi_n * np.sqrt(1 - (1 - c_sigma) ** 2)))
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 10.0)

            # Update covariance matrix
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            C = (1 + c_1 + c_mu - c_1) * C \
                + c_1 * (pc @ pc.T + delta_hsig * C - C) \
                + c_mu * np.sum(weights[:, None, None] * np.outer(X_sorted[:mu] - x_mean, X_sorted[:mu] - x_mean), axis=0) / sigma ** 2

            # Ensure positive definiteness
            C = 0.5 * (C + C.T)
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C = 0.5 * (C + C.T)

            # Update inverse square root
            inv_sqrt_C = self._inv_sqrt(C)

            # Update mean
            x_mean = x_new_mean

        return x_best, f_best

    def _inv_sqrt(self, C):
        """Compute inverse square root of a symmetric positive definite matrix."""
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.clip(eigvals, 1e-12, None)
        inv_sqrt_eigvals = 1.0 / np.sqrt(eigvals)
        return eigvecs @ np.diag(inv_sqrt_eigvals) @ eigvecs.T
# EVOLVE-BLOCK-END
