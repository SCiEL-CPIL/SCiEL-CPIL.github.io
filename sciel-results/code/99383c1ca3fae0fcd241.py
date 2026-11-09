"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f20 (Schwefel) on [-5,5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s kill

        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_pop = max(4, int(4 + 3 * np.log(n)))  # population size
        mu = lambda_pop // 2  # number of parents
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = (2 - mueff / (n + 2) + c_c) / ((n + 1.3) ** 2 + mueff + c_c)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Initial state
        x_mean = rng.uniform(-4, 4, n)
        sigma = 2.0
        C = np.eye(n)
        C_inv = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        step_size = 1.0

        # Track best
        best_x = x_mean.copy()
        best_f = func(x_mean)
        calls = 1

        # Population for first generation
        def make_population():
            nonlocal x_mean, sigma, C, C_inv
            # Generate population using current CMA-ES state
            # Use Cholesky decomposition for sampling
            L = np.linalg.cholesky(C)
            pop = np.zeros((lambda_pop, n))
            for i in range(lambda_pop):
                z = rng.standard_normal(n)
                pop[i] = x_mean + sigma * (L @ z)
                # Clip to bounds
                pop[i] = np.clip(pop[i], -5, 5)
            return pop

        # Evaluate population
        def evaluate(pop):
            nonlocal calls, best_x, best_f
            vals = np.zeros(lambda_pop)
            for i in range(lambda_pop):
                if calls >= budget:
                    break
                vals[i] = func(pop[i])
                calls += 1
                if vals[i] < best_f:
                    best_f = vals[i]
                    best_x = pop[i].copy()
            return vals

        # Main loop
        while calls < budget:
            if time.time() > deadline:
                break

            # Generate and evaluate population
            pop = make_population()
            vals = evaluate(pop)

            if calls >= budget:
                break

            # Sort by fitness
            order = np.argsort(vals)
            selected = pop[order[:mu]]
            selected_vals = vals[order[:mu]]

            # Compute new mean
            new_mean = np.sum(weights[:, None] * selected, axis=0)

            # Update covariance matrix
            # Rank-one update
            y_sigma = (new_mean - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_sigma

            # Rank-mu update
            delta_sig = (np.linalg.norm(p_sigma) / chi_n - 1) * c_sigma * d_sigma
            C = (1 + c_1 * delta_sig + c_mu * (np.sum(weights ** 2) - 2) / mueff) * C
            for i in range(mu):
                y_i = (selected[i] - x_mean) / sigma
                C += c_1 * np.outer(y_sigma, y_sigma)
                C += c_mu * weights[i] * np.outer(y_i, y_i)

            # Update step size
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update correlation matrix
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * y_sigma
            C = (1 + c_1 * delta_sig + c_mu * (np.sum(weights ** 2) - 2) / mueff) * C
            # Regularize C
            C = 0.5 * (C + C.T) + 1e-12 * np.eye(n)
            try:
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                C_inv = np.eye(n)
                C = np.eye(n)

            x_mean = new_mean
            x_mean = np.clip(x_mean, -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
