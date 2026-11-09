"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Diagonal step-size ES with Rank-1 + Rank-mu natural gradient adaptation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 280s safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # Diagonal step sizes (one per dimension) - the key structural change
        sigma = np.ones(dim) * 1.0

        # Parameters for the diagonal CMA-ES variant
        lambda_ = 4 * dim  # population size
        mu = lambda_ // 2  # number of parents

        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w**2)

        # Adaptation rates for diagonal step sizes
        # These control how fast each dimension's step size adapts
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Evolution paths (diagonal version)
        ps = np.zeros(dim)  # step size evolution path
        pc = np.zeros(dim)  # covariance evolution path (diagonal)

        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        while calls < self.budget and time.time() < deadline:
            # Determine how many evaluations we can do in this generation
            remaining = self.budget - calls
            lambda_current = min(lambda_, remaining)

            if lambda_current < 2:
                break

            # Generate population
            ys = []
            fs = []
            for i in range(lambda_current):
                z = rng.standard_normal(dim)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                ys.append(y)
                fs.append(fy)
                calls += 1

                # Update best
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()

                if time.time() > deadline:
                    break

            # Sort by fitness
            if len(ys) < mu:
                break

            idx = np.argsort(fs)[:mu]
            ys_sorted = [ys[i] for i in idx]
            fs_sorted = [fs[i] for i in idx]

            # Update mean
            x_old = x.copy()
            x = sum(w[i] * ys_sorted[i] for i in range(mu))

            # Update evolution path ps for step sizes
            # This adapts the overall scale
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (x - x_old) / sigma

            # Heuristic for sigma update
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**((calls + 1) // lambda_))) < (1.4 + 2 / (dim + 1)) * chi_n

            # Update sigma (overall scale factor applied to all dimensions)
            sigma_scale = np.exp(c_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma * sigma_scale, 1e-10, 10.0)

            # Update evolution path pc for covariance (diagonal)
            pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x - x_old) / sigma

            # Update diagonal step sizes using Rank-1 + Rank-mu updates
            # This is the key: each dimension adapts independently
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)

            # Rank-1 update from pc
            sigma_update = c_1 * np.outer(pc, pc).diagonal()

            # Rank-mu update from weighted outer products
            for i in range(mu):
                diff = (ys_sorted[i] - x_old) / sigma
                sigma_update += c_mu * w[i] * diff**2

            # Combine updates
            sigma = sigma * np.exp((c_1 * delta_hsig - c_1 - c_mu) + sigma_update / sigma**2)
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Ensure sigma stays positive and reasonable
            sigma = np.maximum(sigma, 1e-10)

        return best_x, best_fx
# EVOLVE-BLOCK-END
