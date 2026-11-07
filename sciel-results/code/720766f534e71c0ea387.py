"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES for black-box continuous optimization on BBOB-style problems."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s limit

        rng = np.random.default_rng(42)
        dim = self.dim

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))  # population size
        mu = lambda_ // 2  # number of weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / ((dim + 1.3) ** 2 + mueff)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = (2 - mueff / (dim + 1.8) + c_sigma) / ((dim + 1) ** 2 + mueff)
        c_mu = (2 * (mueff / (dim + 1.8) - c_1)) / ((dim + 1) ** 2 + mueff)
        c_mu = min(c_mu, 2 * (1 - c_1))
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # Initialize
        x_mean = rng.uniform(-3, 3, dim)
        sigma = 2.0
        B = np.eye(dim)
        D = np.ones(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        calls = 0
        best_x = x_mean.copy()
        best_fx = func(x_mean)
        calls += 1

        for iteration in range(self.budget // lambda_ + 1):
            if time.time() > deadline:
                break

            # Generate population
            pop_x = []
            pop_f = []
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                pop_x.append(y)
                fy = func(y)
                calls += 1
                if calls > self.budget:
                    break
                pop_f.append(fy)

            if calls > self.budget:
                break

            if len(pop_f) < lambda_:
                # Pad with best value if we ran out of budget mid-generation
                while len(pop_f) < lambda_:
                    pop_f.append(pop_f[0])
                    pop_x.append(pop_x[0])

            # Sort by fitness
            order = np.argsort(pop_f)
            pop_x = [pop_x[i] for i in order]
            pop_f = [pop_f[i] for i in order]

            # Update best
            if pop_f[0] < best_fx:
                best_fx = pop_f[0]
                best_x = pop_x[0].copy()

            # Compute weighted mean
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * pop_x[i]

            # Update covariance matrix adaptation
            y_vec = (x_new - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_vec
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (iteration + 1)))) / chi_n < 1.4 + 2 / (dim + 1)

            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * y_vec
            delta_heav = (1 - h_sigma) * c_c * (1 - c_c)

            C = B @ (D ** 2) @ B.T
            C = C + (1 - c_1) * 0  # keep C
            # Update C
            C = (1 - c_1 - c_mu) * C
            for i in range(mu):
                y_i = (pop_x[i] - x_mean) / sigma
                C += c_1 * np.outer(y_i, y_i)
                C += weights[i] * np.outer(y_i, y_i)

            # Eigen decomposition for square root
            eigenvalues, B = np.linalg.eigh(C)
            eigenvalues = np.clip(eigenvalues, 1e-20, None)
            D = np.sqrt(eigenvalues)

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = min(sigma, 1e6)
            sigma = max(sigma, 1e-10)

            x_mean = x_new

            # Check convergence
            if np.linalg.norm(x_mean - best_x) < 1e-6 and sigma < 1e-6:
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
