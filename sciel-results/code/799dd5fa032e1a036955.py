"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        d = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = 4 + 3 * int(np.floor(3 * np.log(d)))  # ~14 for d=5
        mu = lambda_ // 2  # number of selected solutions
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Step size control parameters
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = (2 - mu_eff / (d + 1.8) + c_c) / ((d + 1.5) + mu_eff + c_c)
        c_mu = (2 * (mu_eff / (d + 1.8) - 1) + c_1) / ((d + 1.5 + mu_eff) * (1 - c_1) - c_c)
        c_mu = max(0, c_mu)

        # Initial values
        rng = np.random.default_rng(42)
        x_mean = rng.uniform(-5, 5, d)
        sigma = 2.0
        C = np.eye(d)  # Covariance matrix
        pc = np.zeros(d)
        ps = np.zeros(d)
        chi_n = np.sqrt(d) * (1 - 1.0 / (4 * d) + 1.0 / (21 * d ** 2))

        best_x = x_mean.copy()
        best_f = func(best_x)
        calls = 1

        for gen in range(budget - 1):
            # Check time
            if time.time() > deadline:
                break

            # Generate population using current C
            B, D = np.linalg.eigh(C)
            D = np.sqrt(np.clip(np.diag(D), 1e-20, None))

            population = []
            for i in range(lambda_):
                z = rng.standard_normal(d)
                y = x_mean + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                population.append(y)

            # Evaluate
            fitness = []
            for y in population:
                if time.time() > deadline:
                    fitness.append(best_f)
                else:
                    f = func(y)
                    calls += 1
                    fitness.append(f)
                    if f < best_f:
                        best_f = f
                        best_x = y.copy()

            # Sort by fitness
            idx = np.argsort(fitness)
            selected = [population[i] for i in idx[:mu]]

            # Update mean
            x_new = sum(weights[i] * selected[i] for i in range(mu))

            # Update step size
            y_vec = (x_new - x_mean) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_vec
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) / chi_n < 1.4 + 2 / (d + 1)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 1e5)

            # Update covariance
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_vec
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            rank_1 = np.outer(pc, pc)
            rank_mu = sum(weights[i] * np.outer(selected[i] - x_mean, selected[i] - x_mean) / sigma ** 2
                          for i in range(mu))
            C = (1 - c_1 - c_mu + delta_hsig) * C + c_1 * rank_1 + c_mu * rank_mu

            # Ensure symmetry and positive semi-definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-20, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            x_mean = x_new

        return best_x, best_f
# EVOLVE-BLOCK-END
