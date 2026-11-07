"""Optimizer for BBOB-style f13 (Sharp ridge) using a population-based CMA-ES variant."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 320.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)

        # Initial population of size 10
        pop_size = 10
        # Initial sigma: explore the box
        sigma = 2.0
        # Center of search
        c = np.zeros(dim)

        # Best found so far
        best_x = None
        best_f = np.inf

        # Generate initial population
        xs = []
        fs = []
        for _ in range(pop_size):
            x = np.clip(c + sigma * rng.standard_normal(dim), -5, 5)
            f = func(x)
            xs.append(x)
            fs.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        call_count = pop_size

        # CMA-ES parameters
        lambda_ = pop_size
        mu = lambda_ // 2
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Covariance matrix
        C = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        C_sqrt = np.eye(dim)

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Track fitness history for stagnation detection
        last_best = best_f
        no_improve_calls = 0

        while call_count < budget:
            # Time check
            if time.time() > deadline:
                break

            # Sort by fitness
            order = np.argsort(fs)
            xs_sorted = [xs[i] for i in order]
            fs_sorted = [fs[i] for i in order]

            # Update best
            if fs_sorted[0] < best_f:
                best_f = fs_sorted[0]
                best_x = xs_sorted[0].copy()
                no_improve_calls = 0
            else:
                no_improve_calls += 1

            # Compute new center
            c_new = np.sum(w[:, None] * np.array(xs_sorted[:mu]), axis=0)

            # Update evolution paths
            delta_c = (c_new - c) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * delta_c
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * delta_c

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Update covariance
            hs = np.linalg.norm(p_c) / np.sqrt(1 - (1 - c_c) ** (2 * (call_count + 1))) < (1.4 + 2 / (dim + 1)) * chi
            rank_mu = np.outer(delta_c, delta_c)
            if hs:
                rank_one = np.outer(p_c, p_c)
            else:
                rank_one = np.zeros((dim, dim))

            C = (1 + c_c * c_sigma * (2 - c_sigma) * mueff - c_sigma) * C \
                + c_sigma * (c_c * (2 - c_c) * rank_one - rank_mu)
            C = (C + C.T) / 2

            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update center
            c = c_new

            # Generate new population
            xs = []
            fs = []
            for _ in range(pop_size):
                # Sample from N(c, sigma^2 * C)
                z = rng.standard_normal(dim)
                x = np.clip(c + sigma * (eigvecs @ (np.sqrt(eigvals) * z)), -5, 5)
                f = func(x)
                xs.append(x)
                fs.append(f)
                call_count += 1

                if f < best_f:
                    best_f = f
                    best_x = x.copy()

            # Stagnation handling: if no improvement for many iterations, restart with larger exploration
            if no_improve_calls > 50:
                # Restart: keep best point, increase sigma, reset covariance
                c = best_x.copy()
                sigma = max(sigma, 1.0)
                C = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                no_improve_calls = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
