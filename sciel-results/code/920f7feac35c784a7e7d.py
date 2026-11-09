"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive population and deadline guard for rotated Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * log(n))
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2

        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Step size adaptation
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma

        # Covariance matrix adaptation
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = (2 / ((n + 1.3) ** 2 + mu_eff)) * (2 - c_c)
        c_mu = (2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff)) * (
            min(1, c_1) - c_c
        )

        # Initial state
        x = rng.uniform(-3, 3, n)
        fx = func(x)
        sigma = 2.0

        C = np.eye(n)
        C_inv = np.eye(n)
        C_sqrt = np.eye(n)
        C_sqrt_inv = np.eye(n)

        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        call_count = 1

        # Track best
        best_x = x.copy()
        best_f = fx

        def _update_cma(C, C_inv, C_sqrt, C_sqrt_inv, C_inv_sqrt):
            """Update C, C_inv, C_sqrt, C_sqrt_inv, C_inv_sqrt using SVD."""
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            return C, C_inv, C_sqrt, C_sqrt_inv, C_inv_sqrt

        C_inv_sqrt = np.eye(n)

        for i in range(budget - 1):
            if time.time() > deadline - 5:
                break

            # Generate population
            ys = []
            fvals = []
            for j in range(lam):
                z = rng.standard_normal(n)
                y = x + sigma * C_sqrt @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                call_count += 1
                ys.append(y)
                fvals.append(fy)

            # Sort by fitness
            ys = np.array(ys)
            fvals = np.array(fvals)
            order = np.argsort(fvals)
            ys = ys[order]
            fvals = fvals[order]

            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = ys[0].copy()

            # Weighted mean
            y_w = np.zeros(n)
            for j in range(mu):
                y_w += weights[j] * ys[j]

            # Update step size
            v = (C_sqrt_inv @ (y_w - x)) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * v
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)

            # Update evolution path for covariance
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (i + 1)))) < (1.4 + 2 / (n + 1)) * chi_n
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ (y_w - x) / sigma)

            # Update mean
            x = y_w

            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 + c_mu - c_c) * C
            C += c_1 * (np.outer(p_c, p_c) - C)
            outer_sum = np.zeros((n, n))
            for j in range(mu):
                yz = (ys[j] - y_w) / sigma
                outer_sum += weights[j] * np.outer(yz, yz)
            C += c_mu * outer_sum

            # Ensure positive definiteness
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Recompute inverses and square roots periodically
            if i % 10 == 0:
                C, C_inv, C_sqrt, C_sqrt_inv, C_inv_sqrt = _update_cma(C, C_inv, C_sqrt, C_sqrt_inv, C_inv_sqrt)

            # Evaluate new mean if within budget
            if call_count < budget:
                fx_new = func(np.clip(x, -5, 5))
                call_count += 1
                if fx_new < best_f:
                    best_f = fx_new
                    best_x = np.clip(x, -5, 5).copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
