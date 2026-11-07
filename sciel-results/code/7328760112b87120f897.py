"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with deadline guard for BBOB f13 (Sharp ridge)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(dim)))  # population size
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = int(mu / (2 * weights[0]))
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialize
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        best_x = x_mean.copy()
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Initial evaluation
        best_f = evaluate(x_mean)

        # Precompute Cholesky
        L = np.linalg.cholesky(C)

        for gen in range(budget - 1):
            # Check deadline
            if time.time() > deadline:
                break

            # Generate population
            ys = []
            xs = []
            fs = []
            for i in range(lambda_):
                if time.time() > deadline:
                    break
                y = rng.standard_normal(dim)
                x = x_mean + sigma * (C_sqrt_inv @ y)
                x = np.clip(x, -5, 5)
                f = evaluate(x)
                ys.append(y)
                xs.append(x)
                fs.append(f)

            if not ys:
                break

            # Sort by fitness
            idx = np.argsort(fs)
            ys = [ys[i] for i in idx[:mu]]
            xs = [xs[i] for i in idx[:mu]]
            fs = [fs[i] for i in idx[:mu]]

            # Weighted mean
            x_new = sum(weights[i] * xs[i] for i in range(mu))

            # Update step-size control
            y_mean = sum(weights[i] * ys[i] for i in range(mu))
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * C_sqrt_inv @ y_mean
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 1 / dim) * chi_n:
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))

            # Clamp sigma
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update evolution paths and covariance
            pc = (1 - c_c) * pc + c_c * (x_new - x_mean) / sigma
            x_mean_old = x_mean.copy()
            x_mean = x_new

            # Rank-mu update
            delta_sigma = 1 + (1 - c_sigma) * (np.linalg.norm(ps) ** 2 / chi_n ** 2 - 1)
            outer = np.outer(x_new - x_mean_old, x_new - x_mean_old)
            for i in range(mu):
                outer += weights[i] * np.outer((xs[i] - x_mean_old) / sigma, (xs[i] - x_mean_old) / sigma)

            C = (1 + c_1 * delta_sigma + c_mu - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 - c_c) * c_c * (1 - c_c) * C) + c_mu * outer

            # Symmetrize and ensure positive definiteness
            C = (C + C.T) / 2
            try:
                L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                C_sqrt_inv = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                # Fallback: use eigenvalue decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C = (eigvecs * eigvals) @ eigvecs.T
                L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                C_sqrt_inv = np.linalg.inv(L)

        return best_x, best_f
# EVOLVE-BLOCK-END
