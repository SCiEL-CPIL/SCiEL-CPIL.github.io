"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer for BBOB-style f21 (Gallagher 101 peaks)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280
        budget = self.budget
        dim = self.dim
        calls = 0

        def eval_point(x):
            nonlocal calls
            calls += 1
            return func(x)

        # CMA-ES implementation
        # Population size
        lambda_pop = 4 + int(3 * np.log(dim))  # ~14 for dim=5
        # Learning rates
        mu = lambda_pop // 2  # number of weights
        # Weights (logarithmic)
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        # Step size adaptation
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        # Covariance matrix adaptation
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        # Chi-squared approximation for sigma adaptation
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial state
        mean = np.zeros(dim)
        sigma = 2.0  # initial step size (covers most of [-5,5])
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        best_x = mean.copy()
        best_f = np.inf

        # Precompute square root of C
        C_sqrt, _ = np.linalg.slogdet(C)
        C_sqrt = np.linalg.cholesky(C)

        def update_covariance(C, pc, ps, sigma, z, w, mueff, c_c, c_1, c_mu, dim):
            """Update covariance matrix."""
            # Rank-one update
            C += c_1 * (np.outer(pc, pc) + (1 + c_sigma * (1 - c_sigma)) * sigma ** 2 * np.outer(ps, ps) - C)
            # Rank-mu update
            diff = np.array([z[i] for i in range(len(w))])
            C += c_mu * (np.outer(diff, diff).sum(axis=0) - C)
            # Ensure symmetry
            C = (C + C.T) / 2
            # Project to positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            return C

        # Main CMA-ES loop
        while calls < budget and time.time() < deadline:
            # Generate population
            population = []
            for i in range(lambda_pop):
                # Standard normal sample
                z = np.random.standard_normal(dim)
                # Transform using C^0.5
                x = mean + sigma * (C_sqrt @ z)
                x = np.clip(x, -5, 5)
                population.append((x, z))

            # Evaluate population
            fvals = []
            for x, z in population:
                if calls >= budget or time.time() > deadline:
                    break
                f = eval_point(x)
                fvals.append(f)
                if f < best_f:
                    best_f = f
                    best_x = x.copy()

            if len(fvals) < mu:
                break

            # Sort by fitness
            indices = np.argsort(fvals)
            selected = [population[i] for i in indices[:mu]]
            selected_f = [fvals[i] for i in indices[:mu]]

            # Weighted mean of selected
            new_mean = np.zeros(dim)
            for i, (x, z) in enumerate(selected):
                new_mean += weights[i] * x
            mean = new_mean
            mean = np.clip(mean, -5, 5)

            # Update evolution paths
            old_sigma = sigma
            sigma = sigma * np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)

            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / old_sigma * (C_sqrt.T @ (new_mean - mean))
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * (new_mean - mean) / sigma

            # Update covariance matrix
            diff_matrix = np.array([(z - mean) / sigma for _, z in selected])
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - c_c * (2 - c_c)) * np.eye(dim))
            for i in range(mu):
                C += c_mu * weights[i] * np.outer(diff_matrix[i], diff_matrix[i])
            C = (C + C.T) / 2

            # Ensure positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C_sqrt, _ = np.linalg.slogdet(C)
            C_sqrt = np.linalg.cholesky(C)

            # Convergence check
            if np.linalg.norm(sigma * np.linalg.norm(C_sqrt, 2)) < 1e-8:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
