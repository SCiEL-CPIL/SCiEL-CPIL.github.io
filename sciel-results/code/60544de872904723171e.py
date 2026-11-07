"""Optimizer for BBOB-style f11 (Discus) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        # Phase 1: CMA-ES (population-based)
        # Use CMA-ES for robust global search
        from scipy.optimize import minimize

        # Actually, let's implement CMA-ES manually for better control and speed
        # CMA-ES implementation
        rng = np.random.default_rng(42)

        # Initial point: center of the box
        x = np.zeros(dim)
        f_x = func(x)

        # Initial sigma: half the range / sqrt(dim) roughly
        sigma = 2.0 / np.sqrt(dim)

        # Population size
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Step size control parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Covariance matrix
        C = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        C_sqrt = np.eye(dim)

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Track best
        best_x = x.copy()
        best_f = f_x
        call_count = 1

        def update_covariance(ps, C, C_sqrt_inv, C_sqrt, dim, c_1, c_mu, weights, mu, lambda_, mu_eff):
            """Update covariance matrix and its square root."""
            # Covariance update
            rank_one = np.outer(ps, ps)
            rank_mu = np.sum(weights[i] * np.outer(ps[i], ps[i]) for i in range(mu))
            C = (1 + c_1 * (2 - c_1) - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

            # Ensure positive semi-definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update square root using Newton-Schulz or SVD
            # Use SVD for stability
            U, s, Vt = np.linalg.svd(C)
            C_sqrt = U @ np.diag(np.sqrt(s)) @ Vt
            C_sqrt_inv = Vt.T @ np.diag(1.0 / np.maximum(s, 1e-12)) @ U.T

            return C, C_sqrt_inv, C_sqrt

        # Main CMA-ES loop
        while call_count < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate offspring
            offspring = []
            for i in range(lambda_):
                if call_count + 1 > budget:
                    break
                if time.time() > deadline:
                    break

                z = rng.standard_normal(dim)
                y = x + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                f_y = func(y)
                call_count += 1
                offspring.append((f_y, y))

            if len(offspring) == 0:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[0])

            # Update best
            if offspring[0][0] < best_f:
                best_f = offspring[0][0]
                best_x = offspring[0][1].copy()

            # Selection
            selected = [offspring[i][1] for i in range(min(mu, len(offspring)))]
            f_selected = [offspring[i][0] for i in range(min(mu, len(offspring)))]

            # Weighted mean
            x_new = np.sum(weights[:len(selected)] * np.array(selected), axis=0)

            # Check if we made progress
            if np.linalg.norm(x_new - x) < 1e-14:
                # Stagnation: restart with larger sigma or different start
                sigma *= 1.5
                if sigma > 10:
                    break
                continue

            # Update evolution paths
            d_x = x_new - x
            d_x_norm = np.linalg.norm(d_x)

            if d_x_norm > 1e-14:
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * d_x / (sigma * d_x_norm)

                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (call_count + 1 - lambda_))) <
                           (1.4 + 2 / (dim + 1)) * np.sqrt((1 + c_sigma) / (1 - c_sigma)))

                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * d_x / sigma

                # Update covariance
                ps = np.array([w * (xi - x) / sigma for w, xi in zip(weights[:len(selected)], selected)])
                C, C_sqrt_inv, C_sqrt = update_covariance(ps, C, C_sqrt_inv, C_sqrt, dim, c_1, c_mu, weights, mu, lambda_, mu_eff)

            # Update sigma
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-6, 100)

            x = x_new

        # Phase 2: Local refinement with (1+1)-ES around best_x
        if call_count < budget and time.time() < deadline:
            x_local = best_x.copy()
            f_local = best_f
            sigma_local = max(sigma * 0.1, 1e-3)

            while call_count < budget:
                if time.time() > deadline:
                    break

                y = np.clip(x_local + sigma_local * rng.standard_normal(dim), -5, 5)
                f_y = func(y)
                call_count += 1

                if f_y < f_local:
                    x_local, f_local = y, f_y
                    sigma_local *= 1.2
                    best_f = f_local
                    best_x = x_local.copy()
                else:
                    sigma_local *= 0.9
                    if sigma_local < 1e-6:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
