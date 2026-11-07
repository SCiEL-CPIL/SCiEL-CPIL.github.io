"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with time guard for 5D BBOB functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = 4 * int(1 + 3 * dim)  # population size
        mu = lambda_ // 2                # number of recombination weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 - mu_eff / (dim + 1.7) + c_sigma) / ((dim + 2) * (1.8 + c_sigma))
        c_mu = (mu_eff / (dim + 1.7) - 2 + c_sigma) / ((dim + 2) * (1.8 / c_sigma + c_sigma))
        c_mu = min(c_mu, 2 * (1 - c_1))
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # Initial point: try a few starting points, pick the best
        rng = np.random.default_rng(42)

        # Try multiple initial points to handle hidden shifts
        best_x = None
        best_fx = np.inf
        n_init = 10
        for _ in range(n_init):
            x_try = rng.uniform(-4.5, 4.5, dim)
            fx_try = func(x_try)
            if fx_try < best_fx:
                best_fx = fx_try
                best_x = x_try.copy()

        x = best_x.copy()
        fx = best_fx
        calls = n_init

        # Initial step size
        sigma = 2.0

        # CMA-ES state
        B = np.eye(dim)
        D = np.ones(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        steps = 0

        while calls < budget:
            # Time guard
            if time.time() > deadline:
                break

            # Generate population
            offspring = np.zeros((lambda_, dim))
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                offspring[i] = x + sigma * (B @ (D * z))

            # Evaluate offspring
            offspring_f = np.zeros(lambda_)
            for i in range(lambda_):
                y = np.clip(offspring[i], -5, 5)
                offspring_f[i] = func(y)
                calls += 1
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break

            # Sort by fitness
            order = np.argsort(offspring_f)
            offspring = offspring[order]
            offspring_f = offspring_f[order]

            # Update best
            if offspring_f[0] < fx:
                fx = offspring_f[0]
                best_x = offspring[0].copy()

            # Recombination
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * offspring[i]

            # Update covariance matrix adaptation
            y_vec = x_new - x
            sigma_old = sigma

            # Step-size adaptation (1/5 rule via cs)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * (B @ (D * (y_vec / sigma)))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (steps + 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Evolution path for C
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_vec / sigma)

            # Rank-one and rank-mu update
            C = B @ (D ** 2) @ B.T
            delta_c = hsig * (1 - c_c) * (1 + c_c)
            C += (c_1 + c_mu + delta_c) * np.outer(y_vec / sigma, y_vec / sigma)
            for i in range(mu):
                z_i = D * (B.T @ (offspring[i] - x) / sigma)
                C += c_mu * weights[i] * np.outer(z_i, z_i)

            # Eigenvalue decomposition for C
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-12))

            x = x_new
            steps += 1

        return best_x, fx
# EVOLVE-BLOCK-END
