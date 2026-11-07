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
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        max_time = 300.0

        # CMA-ES parameters
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        w = np.log(np.arange(mu, 0, -1)) - np.log(np.arange(lam, mu, -1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        sigma = 2.0
        m = np.zeros(dim)
        C = np.eye(dim)

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 - mu_eff / dim) / (dim + 4)
        c_sigma1 = (1 + 2 / (dim + 1)) * c_sigma
        c_sigma2 = (1 - c_sigma) * min(1, 1 - c_sigma1) * (mu_eff + 2) / (dim + 2)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        best_x = m.copy()
        best_f = func(m)
        calls = 1

        # Eigen decomposition for sampling
        eigval, eigvec = np.linalg.eigh(C)

        while calls + lam <= budget:
            if time.time() - start_time > max_time - 5:
                break

            # Generate population
            y = np.random.randn(lam, dim) @ eigvec.T * np.sqrt(eigval)
            x = m + sigma * y

            f = np.array([func(xi) for xi in x])
            calls += lam

            f_min = f.min()
            if f_min < best_f:
                best_f = f_min
                best_x = x[f == f_min][0].copy()

            idx = np.argsort(f)
            y_sorted = y[idx]

            # Update mean
            y_mean = np.sum(w[:, None] * y_sorted[:mu], axis=0)
            m = m + sigma * y_mean

            # Update paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean

            # Update covariance
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * lam)) <
                       (1.4 + 2 / (dim + 1)) * chi_n)
            C = (1 + c_c * (2 - c_c) * mu_eff) * C + \
                c_c * (2 - c_c) * (p_c @ p_c - C) + \
                (1 - c_c * (2 - c_c) * mu_eff) * (1 - h_sigma) * (C - C) + \
                c_sigma * (2 - c_sigma) * mu_eff * (p_sigma @ p_sigma - C)

            C = 0.5 * (C + C.T)
            eigval, eigvec = np.linalg.eigh(C)
            eigval = np.maximum(eigval, 1e-10)

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

            if np.linalg.norm(p_c) > 4.4 * chi_n:
                p_c = np.zeros(dim)
            if np.linalg.norm(p_sigma) > 6 * chi_n:
                p_sigma = np.zeros(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
