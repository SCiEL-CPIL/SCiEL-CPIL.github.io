"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Rank-1 + Rank-mu Natural Gradient Descent with diagonal step-size vector."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Initialize
        x = rng.uniform(-4, 4, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Diagonal step-size vector (log-sigma per dimension)
        log_sigma = np.full(dim, np.log(1.5))
        sigma = np.exp(log_sigma)

        # Population size
        lam = 12
        mu = lam // 2

        # Success rate for step-size adaptation
        c_sigma = 0.01 + 3.0 / (dim + 3) ** 1.5
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Path accumulators
        p_sigma = np.zeros(dim)
        p_y = np.zeros(dim)

        # Rank-1 and Rank-mu weights for covariance-like update
        # Since we use diagonal, we track per-dimension variance adaptation
        w = np.zeros(mu)
        w_sum = np.sum(1 / np.arange(1, mu + 1) ** 0.5)
        w = (1 / np.arange(1, mu + 1) ** 0.5) / w_sum
        w_sum = np.sum(w)
        w_sum2 = np.sum(w ** 2)
        c_c = 4 * (w_sum2 - 1) / ((dim + 2) ** 2 - 4) + 2 * (w_sum - 1) ** 2 / (dim + 2) ** 2 - 1
        c_1 = 2 / ((dim + 1.3) ** 2 + 1)
        c_mu = min(1 - c_1, 2 * (w_sum2 - 2) / ((dim + 2) ** 2 + 2))

        # Per-dimension variance (diagonal of "covariance")
        diag_var = np.ones(dim)  # initial variance per dimension

        # Stagnation counter for restarts
        stagnation_count = 0
        stagnation_threshold = 50
        restart_count = 0
        max_restarts = 5

        # Track progress for stagnation
        last_improvement_fx = fx
        calls_since_improvement = 0

        for i in range(budget - 1):
            # Check time deadline
            if time.time() > deadline:
                break

            # Generate offspring
            offspring = np.zeros((lam, dim))
            offspring_f = np.zeros(lam)
            for j in range(lam):
                # Sample from N(x, diag(sigma^2 * diag_var))
                z = rng.standard_normal(dim)
                y = x + sigma * np.sqrt(diag_var) * z
                y = np.clip(y, -5, 5)
                offspring[j] = y
                offspring_f[j] = func(y)

                # Check time after each evaluation
                if time.time() > deadline:
                    break

            # Sort offspring by fitness
            idx = np.argsort(offspring_f)
            sorted_offspring = offspring[idx]
            sorted_f = offspring_f[idx]

            # Update best
            if sorted_f[0] < best_fx:
                best_fx = sorted_f[0]
                best_x = sorted_offspring[0].copy()
                last_improvement_fx = best_fx
                calls_since_improvement = 0
            else:
                calls_since_improvement += 1

            # Compute mean of best mu
            y_mean = np.sum(w[:, None] * sorted_offspring[:mu], axis=0)
            d = y_mean - x

            # Success rate for sigma adaptation
            # Use the distance of the mean of best mu to the old mean
            dist = np.linalg.norm(d) / np.sqrt(np.sum(sigma ** 2 * diag_var))
            # Target success rate ~ 1/3
            if dist > 0:
                log_sigma += c_sigma * (dist / chi_n - 1)
            # Clamp log_sigma
            log_sigma = np.clip(log_sigma, np.log(0.01), np.log(10.0))
            sigma = np.exp(log_sigma)

            # Update path accumulators
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * d / np.sqrt(np.sum(sigma ** 2 * diag_var))
            mu_eff = 1.0 / w_sum
            p_y = (1 - c_c) * p_y + np.sqrt(c_c * (2 - c_c) * mu_eff) * d / np.sqrt(np.sum(sigma ** 2 * diag_var))

            # Update diagonal variance (Rank-1 + Rank-mu style)
            # For each dimension, update based on the squared deviation
            # Rank-1 update
            rank1 = np.outer(d, d) / (sigma ** 2)
            # Rank-mu update
            rank_mu = np.zeros((dim, dim))
            for k in range(mu):
                dk = (sorted_offspring[k] - x) / sigma
                rank_mu += w[k] * np.outer(dk, dk)
            rank_mu = (rank_mu - rank1 / w[0])  # avoid double counting

            # Update diagonal variance
            # For diagonal, we only take the diagonal elements
            diag_rank1 = np.diag(rank1)
            diag_rank_mu = np.diag(rank_mu)

            # Combined update
            diag_var = (1 - c_1 - c_mu) * diag_var + c_1 * diag_rank1 + c_mu * diag_rank_mu
            # Ensure positivity
            diag_var = np.maximum(diag_var, 0.01)
            # Normalize to keep mean ~1
            diag_var = diag_var / np.mean(diag_var)

            # Update mean
            x = y_mean
            fx = sorted_f[0]  # best of population

            # Stagnation check
            if calls_since_improvement >= stagnation_threshold:
                stagnation_count += 1
                if stagnation_count >= 3 and restart_count < max_restarts:
                    # Restart from best with larger perturbation
                    restart_count += 1
                    x = best_x + rng.standard_normal(dim) * 2.0
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    log_sigma = np.full(dim, np.log(2.0))
                    sigma = np.exp(log_sigma)
                    diag_var = np.ones(dim)
                    p_sigma = np.zeros(dim)
                    p_y = np.zeros(dim)
                    stagnation_count = 0
                    calls_since_improvement = 0
                else:
                    # Increase exploration
                    log_sigma += 0.3
                    log_sigma = np.clip(log_sigma, np.log(0.01), np.log(10.0))
                    sigma = np.exp(log_sigma)

        return best_x, best_fx
# EVOLVE-BLOCK-END
