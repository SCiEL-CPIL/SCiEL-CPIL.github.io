"""Rewritten optimizer for BBOB f17 (Schaffers F7)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        global_best_x = np.zeros(dim)
        global_best_f = np.inf
        calls = 0

        # CMA-ES parameters
        pop_size = 4 + int(3 * np.log(dim))
        pop_size = max(pop_size, dim + 2)

        # Initialize
        mean = np.random.uniform(-5, 5, dim)
        sigma = 2.0
        C = np.eye(dim)
        C_cholesky = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        cc = (4 + dim / (dim + 4)) / ((dim + 4) + 4 + dim / (dim + 4))
        cs = (2 + 2 / (dim + 1.8)) / (dim + 2 + 2 + 2 / (dim + 1.8))
        c1 = (2 / ((dim + 1.8) ** 2 + 10)) * (1 - cc - cs)
        cmu = min(c1, 2 / ((dim + 2) ** 2 - 2))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Evaluate initial point
        f0 = func(mean)
        calls += 1
        global_best_f = f0
        global_best_x = mean.copy()

        for iteration in range(budget - 1):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            # Generate population
            ys = []
            fs = []
            for i in range(pop_size):
                if calls >= budget or time.time() > deadline:
                    break
                z = np.random.standard_normal(dim)
                y = mean + sigma * (C_cholesky @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                ys.append(y)
                fs.append(fy)

            if len(fs) == 0:
                break

            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = [ys[i] for i in order]
            fs_sorted = [fs[i] for i in order]

            # Update global best
            if fs_sorted[0] < global_best_f:
                global_best_f = fs_sorted[0]
                global_best_x = ys_sorted[0].copy()

            # Update mean
            new_mean = np.zeros(dim)
            for i in range(mu):
                new_mean += weights[i] * ys_sorted[i]
            mean = new_mean

            # Update C and sigma
            diff = np.array([ys_sorted[i] - mean for i in range(mu)])
            delta_mean = (mean - new_mean) / sigma

            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * delta_mean
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (iteration + 1))) / chi_n < 1.4 + 2 / (dim + 1)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * delta_mean

            rank_one = np.outer(pc, pc)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu += weights[i] * np.outer(diff[i], diff[i])

            C = (1 + c1 + cmu - c1) * C + c1 * (rank_one - C) + cmu * rank_mu
            C = (C + C.T) / 2

            try:
                C_cholesky = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                C_cholesky = np.eye(dim)

            sigma = sigma * np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))

            # Clamp sigma
            sigma = np.clip(sigma, 1e-10, 10.0)

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
