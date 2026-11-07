"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts for Rosenbrock-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # 300s limit, 10s safety margin
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        lambda_ = 4 * dim  # population size
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 + 2 / (dim + 1.3)) / ((dim + 1.3 + 2 / mu_eff) * (c_sigma + 1))
        c_mu = 1 - c_1 - 2 * (dim + 1.3) / (dim + mu_eff + 5) * (c_sigma + 1)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        def run_cma_es(x0, sigma0, max_calls):
            x = np.array(x0, dtype=float)
            sigma = sigma0
            B = np.eye(dim)
            D = np.ones(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            calls = 0
            best_x = x.copy()
            best_f = func(x)
            calls += 1
            calls_left = max_calls - calls

            while calls < max_calls and time.time() < deadline:
                # Generate population
                W = np.empty((lambda_, dim))
                for i in range(lambda_):
                    W[i] = B @ (D * rng.standard_normal(dim))
                    calls += 1
                    if calls >= max_calls or time.time() >= deadline:
                        break
                if calls >= max_calls or time.time() >= deadline:
                    break

                # Evaluate
                ys = np.array([x + sigma * W[i] for i in range(lambda_)])
                # Clip to box
                ys = np.clip(ys, -5.0, 5.0)
                fs = np.empty(lambda_)
                for i in range(lambda_):
                    fs[i] = func(ys[i])
                    calls += 1
                    if calls >= max_calls or time.time() >= deadline:
                        break

                if calls >= max_calls or time.time() >= deadline:
                    break

                # Sort by fitness
                order = np.argsort(fs)
                ys_sorted = ys[order]
                fs_sorted = fs[order]

                # Update best
                if fs_sorted[0] < best_f:
                    best_f = fs_sorted[0]
                    best_x = ys_sorted[0].copy()

                # Compute mean
                y_mean = np.sum(weights[:, None] * ys_sorted[:mu], axis=0)

                # Update covariance matrix
                diff = (y_mean - x) / sigma
                D_new = D.copy()
                for i in range(dim):
                    D_new[i] *= np.exp(c_sigma / (2 * dim) * (
                        np.linalg.norm(ps) ** 2 / chi_n ** 2 - 1
                    ))
                D = D_new

                # Update pc
                pc = (1 - c_c) * pc + c_c * diff

                # Update ps
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * B @ D * diff

                # Update B and D via eigendecomposition
                C = (B @ np.diag(D ** 2) @ B.T)
                C = ((1 - c_1 - c_mu) * C
                     + c_1 * (np.outer(pc, pc) + (np.linalg.norm(ps) ** 2 / chi_n ** 2 - 1) * np.eye(dim))
                     + c_mu * np.sum(weights[:, None, None] * np.outer(ys_sorted[:mu] - y_mean, ys_sorted[:mu] - y_mean) * weights[:, None], axis=0) / sigma ** 2)
                # Ensure symmetry
                C = (C + C.T) / 2
                eigenvalues, eigenvectors = np.linalg.eigh(C)
                eigenvalues = np.maximum(eigenvalues, 1e-12)
                B = eigenvectors
                D = np.sqrt(eigenvalues)

                # Update x and sigma
                x = y_mean
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 1e6)

            return best_x, best_f, calls

        # Multi-start strategy
        n_starts = 5
        calls_per_start = budget // n_starts
        best_x = None
        best_f = np.inf
        total_calls = 0

        for start_i in range(n_starts):
            if time.time() >= deadline:
                break
            remaining = budget - total_calls
            if remaining <= 0:
                break

            # Initial point: diverse sampling
            if start_i == 0:
                x0 = np.zeros(dim)
            elif start_i == 1:
                x0 = np.ones(dim) * 2.0
            elif start_i == 2:
                x0 = -np.ones(dim) * 2.0
            elif start_i == 3:
                x0 = rng.uniform(-3, 3, dim)
            else:
                x0 = rng.uniform(-4, 4, dim)

            sigma0 = 2.0
            calls_used = 0
            if start_i < n_starts - 1:
                calls_allowed = calls_per_start
            else:
                calls_allowed = budget - total_calls

            x_res, f_res, calls_used = run_cma_es(x0, sigma0, calls_allowed)
            total_calls += calls_used

            if f_res < best_f:
                best_f = f_res
                best_x = x_res.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
