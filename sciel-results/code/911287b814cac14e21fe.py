"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Global-to-local: uniform sampling + CMA-ES."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # Phase 1: Global exploration with uniform random evaluations
        n_global = min(50, budget // 5)
        best_x = None
        best_f = np.inf
        for _ in range(n_global):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES local optimization
        remaining = budget - n_global
        if remaining <= 0:
            return best_x, best_f

        x = best_x.copy()
        f = best_f
        # Initialize CMA-ES parameters
        sigma = 1.0  # Initial step size
        C = np.eye(n)  # Covariance matrix
        # CMA-ES weights
        mu = max(1, n // 2)
        weights = np.log(np.arange(mu, 0, -1))
        weights /= weights.sum()
        c_sigma = (mu + 2) / (n + 4)
        c_c = (4 + mu / n) / (n + 4 + 2 * mu / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * (sqrt_mu - 2) / ((n + 2) ** 2 + sqrt_mu))
        sqrt_mu = np.sum(weights ** 2) ** 0.5
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (n + 1)) - 1) + c_sigma
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)

        calls_made = n_global
        for _ in range(remaining):
            if time.time() > deadline:
                break

            # Generate lambda candidates
            lam = 10
            y_list = []
            f_list = []
            for _ in range(lam):
                z = rng.standard_normal(n)
                y = np.clip(x + sigma * (C @ z), -5, 5)
                fy = func(y)
                calls_made += 1
                y_list.append(y)
                f_list.append(fy)
                if fy < f:
                    f = fy
                    x = y.copy()

            f_list = np.array(f_list)
            y_list = np.array(y_list)

            # Sort by fitness
            idx = np.argsort(f_list)
            y_sorted = y_list[idx]
            f_sorted = f_list[idx]

            # Update mean
            new_mean = np.sum(weights[:, np.newaxis] * y_sorted[:mu], axis=0)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * sqrt_mu) * (C @ (sigma * (new_mean - x)))
            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) < (1.4 + 2 / (n + 1)) * np.trace(C) / n else 0
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu) * (new_mean - x)
            x = new_mean

            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.sum(weights[:, np.newaxis] * np.outer(y_sorted[:mu] - x, y_sorted[:mu] - x), axis=0)
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Ensure C is positive semi-definite
            C = (C + C.T) / 2

        return x, f
# EVOLVE-BLOCK-END
