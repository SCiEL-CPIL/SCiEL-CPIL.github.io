"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for Rosenbrock optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s limit

        rng = np.random.default_rng(0)
        d = self.dim
        n = self.budget

        # CMA-ES parameters
        lam = 10  # population size
        mu = lam // 2  # number of selected individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (d + mueff + 5)
        c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
        chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        # State
        x_mean = np.zeros(d)
        sigma = 2.0
        B = np.eye(d)
        D = np.ones(d)
        pc = np.zeros(d)
        ps = np.zeros(d)

        best_x = x_mean.copy()
        best_f = np.inf

        # Initial evaluation
        try:
            best_f = func(best_x)
            if not np.isfinite(best_f):
                best_f = 1e10
        except:
            best_f = 1e10

        calls = 1

        while calls < n:
            if time.time() > deadline:
                break

            # Generate population
            population = []
            for i in range(lam):
                z = rng.standard_normal(d)
                y = x_mean + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                population.append(y)

            # Evaluate
            fitness = []
            for y in population:
                if time.time() > deadline:
                    break
                try:
                    f = func(y)
                    if not np.isfinite(f):
                        f = 1e10
                except:
                    f = 1e10
                fitness.append(f)
                calls += 1
                if f < best_f:
                    best_f = f
                    best_x = y.copy()

            if len(fitness) < lam:
                break

            # Sort by fitness
            order = np.argsort(fitness)
            x_selected = np.array([population[i] for i in order[:mu]])

            # Update mean
            x_new_mean = np.sum(weights[:, None] * x_selected, axis=0)

            # Update step size control variables
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / D * (B.T @ (x_new_mean - x_mean)) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * calls)) < (1.4 + 2 / (d + 1)) * chi_d)
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * (x_new_mean - x_mean) / sigma

            # Update covariance matrix
            delta = hsig * (2 - c_c) - 1
            diff = x_selected - x_mean
            C_update = (c_1 + c_mu) * np.outer(pc, pc)
            for i in range(mu):
                C_update += c_mu * weights[i] * np.outer(diff[i], diff[i])
            C = (1 - (c_1 + c_mu + delta * c_1)) * (B @ (D ** 2) @ B.T) + C_update

            # Ensure symmetry
            C = (C + C.T) / 2

            # Eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-20, None)
            D = np.sqrt(eigvals)
            B = eigvecs

            # Update mean and sigma
            x_mean = x_new_mean
            sigma *= np.exp(c_sigma / chi_d * (np.linalg.norm(ps) / chi_d - 1))
            sigma = np.clip(sigma, 1e-10, 100)

        return best_x, best_f
# EVOLVE-BLOCK-END
