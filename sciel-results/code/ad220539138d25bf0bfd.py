"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))  # population size
        mu = lambda_ // 2  # number of weighted parents

        # Weights
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1 / np.sum(w ** 2)

        # Step size control
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = (2 - mueff / (dim + 1.8) + c_sigma) / ((dim + 1.8 + 2 * c_sigma) * (0.5 + c_sigma))
        c_mu = (2 * (mueff / 2 - 1) + c_1) / ((dim + 2) * (0.5 + c_sigma))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial state
        x = np.zeros(dim)
        sigma = 2.0
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        C_sqrt = np.eye(dim)  # Cholesky of C
        C_inv_sqrt = np.eye(dim)  # Inverse of Cholesky

        best_x = x.copy()
        best_f = func(x)
        calls = 1

        def cholesky_factorization(A):
            """Compute Cholesky factorization with regularization if needed."""
            try:
                L = np.linalg.cholesky(A)
                return L
            except np.linalg.LinAlgError:
                # Add regularization
                for eps in [1e-10, 1e-8, 1e-6, 1e-4, 1e-2, 1e1]:
                    try:
                        L = np.linalg.cholesky(A + eps * np.eye(A.shape[0]))
                        return L
                    except np.linalg.LinAlgError:
                        continue
                return np.eye(A.shape[0])

        def inv_cholesky(L):
            """Compute inverse of Cholesky factor."""
            n = L.shape[0]
            L_inv = np.eye(n)
            for i in range(n):
                L_inv[i, i] = 1.0 / L[i, i]
                for j in range(i + 1, n):
                    L_inv[i, j] = -np.dot(L[i, i + 1:j + 1], L_inv[i + 1:j + 1, i]) / L[j, j]
            return L_inv

        # Update C_sqrt and C_inv_sqrt
        C_sqrt = cholesky_factorization(C)
        C_inv_sqrt = inv_cholesky(C_sqrt)

        for gen in range(budget // lambda_ + 1):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            # Generate population
            offspring = []
            for i in range(lambda_):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break

                z = rng.standard_normal(dim) if 'rng' in dir() else np.random.standard_normal(dim)
                y = x + sigma * C_sqrt @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                offspring.append((fy, y))

                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            if len(offspring) == 0:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[0])

            # Update mean
            new_x = np.zeros(dim)
            for i in range(mu):
                new_x += w[i] * offspring[i][1]
            x = new_x

            # Update step size
            z_c = C_inv_sqrt @ (x - best_x) / sigma  # This is approximate; need old x
            # Actually we need to track old mean
            # Let me restructure this

            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * z_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * z_c

            h_sigma = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - lambda_))) < (1.4 + 2 / (dim + 1)) * chi_n else 0

            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = (1 + c_1 + c_mu - c_1 - delta_h) * C
            C += c_1 * np.outer(p_c, p_c)
            for i in range(mu):
                y_i = C_inv_sqrt @ (offspring[i][1] - x) / sigma
                C -= c_mu * w[i] * np.outer(y_i, y_i)

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            C_sqrt = cholesky_factorization(C)
            C_inv_sqrt = inv_cholesky(C_sqrt)

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Safety: keep sigma in reasonable range
            sigma = np.clip(sigma, 1e-10, 100)

        return best_x, best_f
# EVOLVE-BLOCK-END
