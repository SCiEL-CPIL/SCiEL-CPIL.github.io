"""Rewritten optimizer for BBOB-style suite task."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive population, restarts, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf

        # Use a deterministic seed for reproducibility
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))  # population size
        mu = lambda_ // 2  # number of selected individuals
        weights = np.log(np.arange(mu, 0, -1)) - np.log(np.arange(mu, 0, -1)).sum()
        weights /= weights.sum()
        mueff = 1 / np.sum(weights ** 2)
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Number of restarts
        n_restarts = 3
        budget_per_restart = max(1, budget // n_restarts)

        for restart in range(n_restarts):
            if time.time() > deadline:
                break

            # Initialize
            if best_x is None:
                x0 = rng.uniform(-4, 4, dim)
            else:
                # Perturb the best solution found so far
                x0 = best_x + 0.5 * rng.standard_normal(dim)
                x0 = np.clip(x0, -5, 5)

            x_mean = x0.copy()
            sigma = 2.0
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            B = np.eye(dim)
            D = np.ones(dim)
            C = np.eye(dim)
            calls_in_restart = 0

            def sample():
                nonlocal calls_in_restart
                z = rng.standard_normal((lambda_, dim))
                # Transform using Cholesky of C
                # C = B @ diag(D^2) @ B.T
                # x = x_mean + sigma * z @ (B @ diag(D))
                transformed = z @ (B * D)  # (lambda_, dim)
                ys = x_mean + sigma * transformed
                ys = np.clip(ys, -5, 5)
                return ys

            def evaluate(ys):
                nonlocal calls_in_restart
                fs = np.array([func(y) for y in ys])
                calls_in_restart += lambda_
                return fs

            # Initial evaluation
            ys = sample()
            fs = evaluate(ys)
            if calls_in_restart > budget_per_restart:
                # Update global best
                idx = np.argmin(fs)
                if fs[idx] < best_f:
                    best_f = fs[idx]
                    best_x = ys[idx].copy()
                continue

            # Main loop
            while calls_in_restart < budget_per_restart:
                if time.time() > deadline:
                    break

                # Sort by fitness
                order = np.argsort(fs)
                y_parents = ys[order[:mu]]
                f_parents = fs[order[:mu]]

                # Update mean
                x_mean_old = x_mean.copy()
                x_mean = np.sum(weights[:, None] * y_parents, axis=0)

                # Update step size control
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / chi * (B @ (D * (np.sum(weights[:, None] * y_parents, axis=0) - x_mean_old) / sigma))
                pc = (1 - cc) * pc + cc / np.sqrt(cc * (2 - cc)) * (x_mean - x_mean_old) / sigma

                # Update covariance
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_in_restart + 1) // lambda_)) < (1.4 + 2 / (dim + 1)) * chi:
                    rank_ps = np.outer(ps, ps)
                else:
                    rank_ps = np.zeros((dim, dim))
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    zi = (y_parents[i] - x_mean_old) / sigma
                    rank_mu += weights[i] * np.outer(zi, zi)
                C = (1 + c1 * (rank_ps - np.eye(dim)) + cmu * rank_mu - c1 * np.eye(dim)) * (1 - c1) + c1 * np.eye(dim) + cmu * rank_mu
                # Simpler and more stable:
                C = (1 - c1 - cmu) * C + c1 * (rank_ps + np.eye(dim)) + cmu * rank_mu

                # Ensure symmetry
                C = (C + C.T) / 2

                # Eigen decomposition
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-20))
                C = (B * D) @ B.T

                # Update sigma
                sigma *= np.exp(cs / (5 * dim) * (np.linalg.norm(ps) / chi - 1))
                sigma = np.clip(sigma, 1e-10, 10)

                # Sample next generation
                ys = sample()
                fs = evaluate(ys)

                # Update global best
                idx = np.argmin(fs)
                if fs[idx] < best_f:
                    best_f = fs[idx]
                    best_x = ys[idx].copy()

                if calls_in_restart >= budget_per_restart:
                    break

        # Final polish: a few local steps around the best solution
        if best_x is not None and time.time() < deadline - 10:
            sigma_small = 0.1
            for _ in range(min(20, budget - (budget_per_restart * min(n_restarts, 3)))):
                if time.time() > deadline:
                    break
                y = best_x + sigma_small * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                if fy < best_f:
                    best_f = fy
                    best_x = y
                    sigma_small *= 0.5
                else:
                    sigma_small *= 0.7

        return best_x, best_f
# EVOLVE-BLOCK-END
