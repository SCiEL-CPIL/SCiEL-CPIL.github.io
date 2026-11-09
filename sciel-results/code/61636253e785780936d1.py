"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population-based selection and covariance adaptation."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # Safety margin
        n = self.dim
        budget = self.budget

        best_x = np.zeros(n)
        best_f = np.inf
        call_count = 0
        rng = np.random.default_rng(12345)

        def elapsed():
            return time.time() - t0

        def timed_out():
            return elapsed() > deadline

        # CMA-ES parameters
        lambda_ = 8  # population size
        mu = lambda_ // 2  # number of selected parents
        # Weights for (lambda/2)-selection
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        # Initial state
        x_mean = np.zeros(n)
        sigma = 1.0
        C = np.eye(n)
        C_inv = np.eye(n)
        C_sqrt = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Evaluate initial point
        fx = func(x_mean)
        call_count += 1
        if fx < best_f:
            best_f = fx
            best_x = x_mean.copy()

        while call_count < budget and not timed_out():
            # Generate population
            offspring = []
            offspring_f = []
            for i in range(lambda_):
                if call_count >= budget or timed_out():
                    break
                z = rng.standard_normal(n)
                y = np.clip(x_mean + sigma * (C_sqrt @ z), -5, 5)
                fy = func(y)
                call_count += 1
                offspring.append(y)
                offspring_f.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if len(offspring) < mu:
                break

            # Sort by fitness
            idx = np.argsort(offspring_f)
            selected = [offspring[i] for i in idx[:mu]]
            selected_f = [offspring_f[i] for i in idx[:mu]]

            # Update mean
            old_mean = x_mean.copy()
            x_mean = sum(w * s for w, s in zip(weights, selected))

            # Update CMA-ES state
            y_vec = (x_mean - old_mean) / sigma if sigma > 0 else np.zeros(n)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv @ y_vec)
            norm_p_sigma = np.linalg.norm(p_sigma)

            if norm_p_sigma / chi_n < (1.4 + 1 / n) * np.sqrt((1 - c_sigma) / (1 - c_sigma ** (2 * (call_count // lambda_ + 1)))):
                p_c = (1 - c_c) * p_c + c_c * y_vec

            # Update covariance
            delta_diag = (norm_p_sigma / chi_n > (1.4 + 1 / n)) * (1 - c_c)
            rank_one = np.outer(p_c, p_c)
            rank_mu = sum(w * np.outer(s - old_mean, s - old_mean) / (sigma ** 2) for w, s in zip(weights, selected))
            C = (1 + c_1 * delta_diag - c_1 - c_mu) * C + c_1 * (rank_one + (1 - delta_diag) * np.eye(n)) + c_mu * rank_mu

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            # Regularize
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update inverse and sqrt
            C_inv = np.linalg.inv(C)
            # Compute matrix square root via eigenvalue decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Restart if sigma is too small or too large
            if sigma < 1e-8 or sigma > 10.0:
                # Restart with a new random point
                x_mean = rng.uniform(-4, 4, n)
                sigma = 1.0
                C = np.eye(n)
                C_inv = np.eye(n)
                C_sqrt = np.eye(n)
                p_sigma = np.zeros(n)
                p_c = np.zeros(n)
                fx = func(x_mean)
                call_count += 1
                if fx < best_f:
                    best_f = fx
                    best_x = x_mean.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
