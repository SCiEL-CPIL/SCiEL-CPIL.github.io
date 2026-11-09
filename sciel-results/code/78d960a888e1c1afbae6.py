"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive sigma and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 300s limit

        best_x = None
        best_f = np.inf

        # Multiple restarts with different seeds
        # CMA-ES with population size ~ 4 + 3*ln(n)
        lambda_ = 4 + int(3 * np.log(dim))
        # For dim=5, lambda_ = 4 + 3*1.609 = 4 + 4 = 8
        lambda_ = max(lambda_, 4)

        # Number of evaluations per restart
        # We'll use a fraction of budget per restart
        n_restarts = 3
        evals_per_restart = budget // n_restarts

        # Start with a random point
        rng = np.random.default_rng(42)
        x0 = rng.uniform(-5, 5, dim)
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0
        calls = 1

        for restart in range(n_restarts):
            if calls >= budget:
                break
            if time.time() > deadline:
                break

            # Initialize CMA-ES
            if restart == 0:
                x_mean = x0.copy()
            else:
                # Restart from a new random point near best or random
                if rng.random() < 0.5:
                    x_mean = best_x + rng.normal(0, 1.0, dim)
                else:
                    x_mean = rng.uniform(-5, 5, dim)
                x_mean = np.clip(x_mean, -5, 5)
                # Evaluate starting point
                fx_mean = func(x_mean)
                calls += 1
                if fx_mean < best_f:
                    best_f = fx_mean
                    best_x = x_mean.copy()

            sigma = 1.0
            # Covariance matrix C = I
            C = np.eye(dim)
            # Eigen decomposition for sampling
            D = np.sqrt(np.linalg.eigvalsh(C))
            B = np.eye(dim)  # eigenvectors (identity for I)

            # CMA-ES parameters
            mu = lambda_ // 2
            # Weights
            w = np.log(mu + 0.5) - np.log(np.arange(1, lambda_ + 1))
            w /= np.sum(w)
            mu_eff = 1.0 / np.sum(w ** 2)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

            ps = np.zeros(dim)
            pc = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            for gen in range(evals_per_restart):
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break

                # Generate lambda_ samples
                ys = np.zeros((lambda_, dim))
                xs = np.zeros((lambda_, dim))
                fs = np.zeros(lambda_)

                for i in range(lambda_):
                    if time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    y = B @ (D * z)
                    x_new = np.clip(x_mean + sigma * y, -5, 5)
                    ys[i] = y
                    xs[i] = x_new
                    fs[i] = func(x_new)
                    calls += 1

                    if fs[i] < best_f:
                        best_f = fs[i]
                        best_x = x_new.copy()

                if calls >= budget:
                    break

                # Sort by fitness
                idx = np.argsort(fs)
                ys = ys[idx]
                xs = xs[idx]
                fs = fs[idx]

                # Update mean
                x_mean_new = np.sum(w[:, None] * xs[:mu], axis=0)
                x_mean_new = np.clip(x_mean_new, -5, 5)

                # Update evolution paths
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D * (w @ ys[:mu])))
                h_sigma = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
                pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean_new - x_mean) / sigma
                x_mean = x_mean_new

                # Update covariance
                delta = h_sigma * (1 / c_c - 1) + 1
                C = (1 + c_1 * (delta - 1) - c_1 - c_mu) * C
                C += c_1 * np.outer(pc, pc)
                C += c_mu / sigma ** 2 * (w @ (ys[:mu] * (ys[:mu] @ (B.T @ B @ ys[:mu].T))))
                # Simplified: use rank-mu update
                v = (xs[:mu] - x_mean) / sigma
                C += c_mu / sigma ** 2 * (w @ (v * (v @ (B.T @ B @ v.T))))
                # Ensure symmetry
                C = (C + C.T) / 2

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Re-derive eigen decomposition periodically
                if gen % 5 == 0:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    D = np.sqrt(eigvals)
                    B = eigvecs

        return best_x, best_f
# EVOLVE-BLOCK-END
