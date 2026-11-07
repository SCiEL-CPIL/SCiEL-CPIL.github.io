"""Optimizer using CMA-ES for BBOB f23 (Katsuura)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with deadline guard for continuous optimization on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * log(dim + 1))
        lam = 4 + int(3 * np.log(dim + 1))
        # Step size
        sigma = 1.0
        # Initial mean: center of box
        x_mean = np.zeros(dim)
        # Initial covariance: identity
        C = np.eye(dim)
        # Precompute Cholesky of C
        try:
            C_sqrt = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C_sqrt = np.eye(dim)

        # Evolution path weights
        mu = lam // 2  # number of recombination weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / dim) / ((dim + 2) ** 2 + mu_eff))
        # Damping for sigma update
        d_sigma = c_sigma + 2 * min(1, (mu_eff - 1) / (dim + 1)) * (1 - c_sigma)
        # Threshold for h_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # State
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        best_x = None
        best_f = np.inf
        calls_made = 0

        def clip_to_box(x):
            return np.clip(x, -5.0, 5.0)

        # Initial evaluation of mean
        x0 = clip_to_box(x_mean)
        f0 = func(x0)
        calls_made += 1
        best_x = x0.copy()
        best_f = f0

        for gen in range(budget // lam + 1):
            if calls_made >= budget:
                break
            if time.time() > deadline:
                break

            # Generate population
            ys = np.zeros((lam, dim))
            for i in range(lam):
                z = np.random.standard_normal(dim)
                y = x_mean + sigma * (C_sqrt @ z)
                ys[i] = y

            # Evaluate population (respecting budget and deadline)
            fs = np.zeros(lam)
            for i in range(lam):
                if calls_made >= budget:
                    break
                if time.time() > deadline:
                    break
                x_i = clip_to_box(ys[i])
                f_i = func(x_i)
                fs[i] = f_i
                calls_made += 1
                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()

            # If we ran out of budget or time, break
            if calls_made >= budget or time.time() > deadline:
                break

            # Sort by fitness
            order = np.argsort(fs)
            ys = ys[order]
            fs = fs[order]

            # Recombination
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * ys[i]

            # Update covariance matrix adaptation
            y_vec = (x_new - x_mean) / sigma
            C_old = C.copy()
            C += -c_1 * (C + sigma ** 2 * C) + c_mu * np.outer(y_vec, y_vec)

            # Update evolution paths
            p_c = (1 - c_c) * p_c + c_c * y_vec
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_vec

            # h_sigma: check if p_sigma is reasonable
            h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made // lam + 1))) < (1.4 + 2 / (dim + 1)) * chi_n
            if h_sigma:
                p_c = (1 - c_c) * p_c + c_c * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_vec

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 1e6)

            # Update covariance matrix
            if h_sigma:
                C += -c_1 * (C + sigma ** 2 * C) + c_mu * np.outer(y_vec, y_vec)
            else:
                C += -c_1 * C + c_mu * np.outer(y_vec, y_vec)

            # Ensure C is symmetric and positive definite
            C = 0.5 * (C + C.T)
            # Add small regularization
            C += 1e-12 * np.eye(dim)
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If Cholesky fails, fall back to identity
                C = np.eye(dim)
                C_sqrt = np.eye(dim)

            # Update mean
            x_mean = x_new

            # Clip mean to box
            x_mean = clip_to_box(x_mean)

        return best_x, best_f
# EVOLVE-BLOCK-END
