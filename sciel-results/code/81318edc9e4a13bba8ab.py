"""CMA-ES optimizer for BBOB f14 (Different powers)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f14 (Different powers)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin for 330s wall-clock limit
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf

        def time_ok():
            return time.time() < deadline

        def budget_ok():
            return calls < budget

        def record(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # CMA-ES parameters
        # Population size: 4 + floor(3 * log(dim))
        pop_size = 4 + int(3 * np.log(dim))
        if pop_size < 4:
            pop_size = 4

        # Initial standard deviation
        sigma = 1.0

        # Initial mean: center of the box
        mean = np.zeros(dim)

        # Covariance matrix: identity
        C = np.eye(dim)

        # Inverse of covariance matrix
        C_inv = np.eye(dim)

        # Cholesky factor of C
        try:
            C_chol = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C_chol = np.eye(dim)

        # Step size control
        c_sigma = (4 + dim / 3) / (dim + 4 + 2 * dim / 3)
        d_sigma = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (dim + 1)) - 1) + c_sigma

        # Covariance matrix adaptation
        c_c = (4 + dim / dim) / (dim + 4 + 3 * dim / dim)
        mu_eff = pop_size / 2  # Effective population size
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_mu = (2 * (mu_eff - 2 + 2 / mu_eff)) / ((dim + 2) ** 2 + mu_eff)

        # Weights for evolution path
        weights = np.log(pop_size / 2 + 0.5) - np.log(np.arange(1, pop_size + 1))
        weights = weights / np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Evaluate initial population
        population = []
        for i in range(pop_size):
            if not time_ok() or not budget_ok():
                break
            z = np.random.randn(dim)
            x = mean + sigma * (C_chol @ z)
            x = np.clip(x, -5, 5)
            f = func(x)
            calls += 1
            population.append((x, f))
            record(x, f)

        if not population:
            return best_x, best_f

        # Main CMA-ES loop
        while time_ok() and budget_ok():
            # Sort population by fitness
            population.sort(key=lambda x: x[1])

            # Update best
            best_x_curr = population[0][0]
            best_f_curr = population[0][1]
            if best_f_curr < best_f:
                best_f = best_f_curr
                best_x = best_x_curr.copy()

            # Check convergence
            if len(population) < 2:
                break
            f_spread = population[-1][1] - population[0][1]
            if f_spread < 1e-12:
                break

            # Compute weighted mean
            new_mean = np.zeros(dim)
            for i in range(min(pop_size, len(population))):
                new_mean += weights[i] * population[i][0]

            # Update evolution paths
            old_mean = mean
            mean = new_mean

            # Update sigma (step size)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_chol @ ((mean - old_mean) / sigma))
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** 2) / np.sqrt(1 - (1 - c_sigma) ** (2 * int(np.ceil(1 / c_sigma))))) < (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (dim + 1))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * int(np.ceil(1 / c_sigma)))) - 1))

            # Update covariance matrix
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (mean - old_mean) / sigma

            # Rank-one update
            diff = (mean - old_mean) / sigma
            C = (1 + c_c * (2 - c_c) - c_mu) * C + c_mu * np.outer(diff, diff)

            # Rank-mu update
            rank_mu = np.zeros((dim, dim))
            for i in range(min(pop_size, len(population))):
                y = (population[i][0] - old_mean) / sigma
                rank_mu += weights[i] * np.outer(y, y)
            C += (1 + c_c * (2 - c_c) - c_mu - c_mu) * rank_mu

            # Ensure symmetry
            C = (C + C.T) / 2

            # Project to positive definite
            try:
                C_inv = np.linalg.inv(C)
                C_chol = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback: reset covariance
                C = np.eye(dim)
                C_inv = np.eye(dim)
                C_chol = np.eye(dim)
                p_c = np.zeros(dim)
                p_sigma = np.zeros(dim)

            # Generate new population
            population = []
            for i in range(pop_size):
                if not time_ok() or not budget_ok():
                    break
                z = np.random.randn(dim)
                x = mean + sigma * (C_chol @ z)
                x = np.clip(x, -5, 5)
                f = func(x)
                calls += 1
                population.append((x, f))
                record(x, f)

        return best_x, best_f
# EVOLVE-BLOCK-END
