"""Rewritten optimizer: multi-start CMA-ES with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES for black-box minimization on [-5,5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = np.inf

        # Multi-start: try several different starting points
        n_restarts = max(1, min(10, budget // 50))
        evals_per_restart = max(30, budget // n_restarts)

        for r in range(n_restarts):
            if time.time() > deadline:
                break

            # Choose starting point
            if r == 0:
                x0 = np.zeros(dim)
                sigma0 = 2.5
            else:
                x0 = rng.uniform(-3.5, 3.5, dim)
                sigma0 = 1.5

            # CMA-ES setup
            lam = max(4, int(4 + 2 * np.log(dim)))
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights**2)
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

            x_mean = x0.copy()
            sigma = sigma0
            B = np.eye(dim)
            D = np.ones(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            C = np.eye(dim)

            evals_used = 0
            max_evals = min(evals_per_restart, budget - (r * evals_per_restart))

            for gen in range(max_evals):
                if time.time() > deadline:
                    break
                if evals_used >= max_evals:
                    break

                # Generate population
                ys = []
                for i in range(lam):
                    if time.time() > deadline:
                        break
                    z = rng.standard_normal(dim)
                    y = x_mean + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    evals_used += 1
                    ys.append((y, fy))

                if len(ys) < lam:
                    break

                # Sort by fitness
                ys.sort(key=lambda t: t[1])

                # Update best
                if ys[0][1] < best_f:
                    best_f = ys[0][1]
                    best_x = ys[0][0].copy()

                # Compute weighted mean (selection mean)
                w_mean = np.zeros(dim)
                for i in range(mu):
                    w_mean += weights[i] * ys[i][0]

                # Update sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (B @ (D * ((w_mean - x_mean) / sigma)))
                if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(gen + 1)) < (1.4 + 2 / (dim + 1)) * chi_n:
                    sigma *= np.exp(c_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

                # Save old mean for C update
                x_mean_old = x_mean.copy()
                x_mean = np.clip(w_mean, -5, 5)

                # Update C matrix
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) - C)
                for i in range(mu):
                    diff = (ys[i][0] - x_mean_old) / sigma
                    C += c_mu * weights[i] * np.outer(diff, diff)

                # Update p_c
                diff_mean = (x_mean - x_mean_old) / sigma
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * diff_mean

                # Recompute B, D from C
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                D = np.sqrt(eigvals)
                B = eigvecs
                C = (C + C.T) / 2

        return best_x, best_f
# EVOLVE-BLOCK-END
