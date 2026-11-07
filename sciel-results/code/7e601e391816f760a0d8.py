"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for Rosenbrock with adaptive population and covariance updates."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initialize with random points to find a good starting center
        candidates = []
        for _ in range(10):
            x_try = rng.uniform(-4, 4, n)
            f_try = func(x_try)
            candidates.append((f_try, x_try))
        candidates.sort(key=lambda t: t[0])
        x_mean = candidates[0][1].copy()
        f_best = candidates[0][0]
        x_best = x_mean.copy()
        calls_made = 10

        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(n)))
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 1)) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(n * (1 - 1 / (4 * n) + 1 / (21 * n ** 2)))

        sigma = 2.0
        C = np.eye(n)
        C_inv = np.eye(n)
        C_sqrt = np.eye(n)
        C_sqrt_inv = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        # Helper to update square root decomposition
        def update_sqrt(C):
            """Compute square root and inverse square root of C."""
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
            return C_sqrt, C_sqrt_inv, C_inv

        C_sqrt, C_sqrt_inv, C_inv = update_sqrt(C)

        while calls_made < budget:
            if time.time() > deadline - 5.0:
                break

            # Generate population
            z_matrix = rng.standard_normal((lam, n))
            y_matrix = x_mean + sigma * (C_sqrt @ z_matrix.T).T
            y_matrix = np.clip(y_matrix, -5, 5)

            # Evaluate
            f_values = []
            for i in range(lam):
                if calls_made >= budget:
                    break
                fy = func(y_matrix[i])
                f_values.append(fy)
                calls_made += 1

            if len(f_values) < mu:
                break

            # Sort by fitness
            indices = np.argsort(f_values)
            selected_y = y_matrix[indices[:mu]]
            selected_f = f_values[indices[:mu]]

            # Update best
            if selected_f[0] < f_best:
                f_best = selected_f[0]
                x_best = selected_y[0].copy()

            # Compute new mean
            x_mean_new = np.zeros(n)
            for i in range(mu):
                x_mean_new += weights[i] * selected_y[i]

            # Update evolution paths
            diff = (x_mean_new - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(
                c_sigma * (2 - c_sigma) * mu_eff
            ) * diff
            p_c = (1 - c_c) * p_c + np.sqrt(
                c_c * (2 - c_c) * mu_eff
            ) * (C_sqrt_inv @ diff)

            # Update sigma
            sigma *= np.exp(
                c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1.0)
            )
            sigma = np.clip(sigma, 1e-8, 100.0)

            # Update covariance matrix
            h_sigma = (
                np.linalg.norm(p_sigma)
                / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made + 1) / (n + mu_eff)))
                < (1.4 + 2 / (n + 1)) * chi_n
            )

            C += c_1 * (
                np.outer(p_c, p_c)
                - (1 - (1 - c_c) ** 2) * C
            ) + c_mu * (
                np.sum(weights[:, None, None] * np.outer(selected_y - x_mean, selected_y - x_mean), axis=0)
                / (sigma ** 2)
                - (1 - (1 - c_mu) ** 2) * C
            )

            # Ensure positive semi-definiteness
            C = (C + C.T) / 2
            eigvals = np.linalg.eigvalsh(C)
            if np.any(eigvals < 1e-10):
                C += (1e-10 - np.min(eigvals)) * np.eye(n)

            C_sqrt, C_sqrt_inv, C_inv = update_sqrt(C)
            x_mean = x_mean_new

        return x_best, f_best
# EVOLVE-BLOCK-END
