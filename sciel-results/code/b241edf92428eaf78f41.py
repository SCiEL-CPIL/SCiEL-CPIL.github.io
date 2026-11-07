"""Optimized for BBOB f12 (Bent Cigar) with hidden affine transforms."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES population-based optimizer with adaptive step size."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 280.0

        # CMA-ES parameters
        lam = 4 + int(3 * np.log(dim))  # ~8 for dim=5
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # State
        cm = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-12)
        D = 1.0 / np.sqrt(eigvals)
        B = eigvecs

        best_x = cm.copy()
        best_f = func(cm)
        calls_made = 1
        gen = 0

        while calls_made < budget:
            if time.time() > deadline:
                break

            gen += 1
            # Generate offspring
            ys = []
            for _ in range(lam):
                if calls_made >= budget:
                    break
                z = np.random.standard_normal(dim)
                y = cm + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                ys.append(y)
                calls_made += 1

            if len(ys) < mu:
                break

            # Evaluate
            fs = []
            for y in ys:
                fy = func(y)
                fs.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

            # Sort
            order = np.argsort(fs)
            ys = [ys[i] for i in order[:mu]]

            # Weighted recombination
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * (ys[i] - cm) / sigma

            # Update paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_w
            p_sigma_norm = np.linalg.norm(p_sigma)
            h_sigma = (p_sigma_norm / np.sqrt(1 - (1 - c_sigma) ** (2 * gen))) < (1.4 + 2 / (dim + 1)) * chi_n

            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * y_w

            # Update mean
            cm_old = cm.copy()
            cm = cm_old.copy()
            for i in range(mu):
                cm += weights[i] * ys[i]
            cm = np.clip(cm, -5, 5)

            # Update covariance
            C = (1 - c_1 - c_mu) * C
            if h_sigma:
                C += c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C)
            else:
                C += c_1 * np.outer(p_c, p_c)
            for i in range(mu):
                diff = ys[i] - cm_old
                C += c_mu * weights[i] * np.outer(diff, diff)

            C = (C + C.T) / 2 + 1e-14 * np.eye(dim)

            # Eigen decomposition
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-14)
                D = 1.0 / np.sqrt(eigvals)
                B = eigvecs
            except np.linalg.LinAlgError:
                break

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (p_sigma_norm / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
