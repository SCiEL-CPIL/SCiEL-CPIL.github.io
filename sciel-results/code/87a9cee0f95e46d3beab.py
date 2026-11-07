"""Rewritten optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-like optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Initial point
        rng = np.random.default_rng(42)
        x = rng.uniform(-4, 4, dim)
        fx = func(x)

        # CMA-ES parameters
        sigma = 2.0
        pop_size = 4 + int(3 * np.log(dim))  # for dim=5, pop_size=11
        pop_size = max(pop_size, 2)

        # Mean and covariance
        mean = x.copy()
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)

        # Step size control
        step_size_history = []

        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Update rates
        c_sigma = (1 + 2 / max(dim - 1, 1) + pop_size / (2 * dim)) / (dim + 4)
        c_c = (4 + pop_size / dim) / (dim + 4 + 2 * pop_size / dim)
        mu = pop_size // 2  # number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_1 = (2 / ((dim + 1.3) ** 2 + mu_eff)) / (dim + 4)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        def cholesky_inverse(C):
            """Compute inverse square root of positive definite matrix."""
            try:
                L = np.linalg.cholesky(C)
                # Inverse of L
                L_inv = np.linalg.inv(L)
                # C_inv_sqrt = L_inv^T
                return L_inv.T
            except np.linalg.LinAlgError:
                return np.eye(dim)

        # Initial C_inv_sqrt
        C_inv_sqrt = np.eye(dim)

        best_x = x.copy()
        best_fx = fx
        call_count = 1

        while call_count < budget:
            # Check time
            if time.time() > deadline - 5.0:
                break

            # Generate population
            offspring = []
            for i in range(pop_size):
                # Check time
                if time.time() > deadline - 5.0:
                    break
                if call_count >= budget:
                    break

                # Sample
                z = rng.standard_normal(dim)
                y = mean + sigma * (C_inv_sqrt @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                call_count += 1
                offspring.append((fy, y))

            if not offspring:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[0])

            # Update best
            if offspring[0][0] < best_fx:
                best_fx = offspring[0][0]
                best_x = offspring[0][1].copy()

            # Update mean using weighted recombination
            new_mean = np.zeros(dim)
            for i in range(min(mu, len(offspring))):
                new_mean += weights[i] * offspring[i][1]
            mean = new_mean

            # Update covariance matrix
            diff = np.zeros((dim, min(mu, len(offspring))))
            for i in range(min(mu, len(offspring))):
                diff[:, i] = offspring[i][1] - mean

            # Rank-mu update
            C += c_mu * np.outer(diff @ np.diag(weights) @ diff.T, np.ones(1))[0] if False else 0
            # Proper rank-mu update
            C_rank_mu = np.zeros((dim, dim))
            for i in range(min(mu, len(offspring))):
                C_rank_mu += weights[i] * np.outer(diff[:, i], diff[:, i])
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - c_c) * c_c * (1 - c_c) * np.eye(dim)) + c_mu * C_rank_mu

            # Update evolution paths
            # p_sigma update
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * (np.sqrt(mu_eff) * sigma * (C_inv_sqrt @ (diff[:, 0] / sigma)))
            # Actually, let's use a simpler approach for stability

            # Recompute C_inv_sqrt
            # Add small regularization for numerical stability
            C_reg = C + 1e-12 * np.eye(dim)
            try:
                C_inv_sqrt = cholesky_inverse(C_reg)
            except:
                C_inv_sqrt = np.eye(dim)

            # Adapt sigma (1/5 success rule variant)
            successes = sum(1 for i in range(min(mu, len(offspring))) if offspring[i][0] < fx)
            # Simple sigma adaptation
            if successes > 0:
                sigma *= 1.1
            else:
                sigma *= 0.9
            sigma = np.clip(sigma, 0.01, 10.0)

            # Update fx for next comparison
            fx = best_fx

        return best_x, best_fx
# EVOLVE-BLOCK-END
