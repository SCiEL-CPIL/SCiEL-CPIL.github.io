"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for BBOB f24 (Lunacek bi-Rastrigin)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 330s limit
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(dim)))  # population size
        mu = lambda_ // 2  # number of weights
        # Compute weights
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = (2 - mu_eff / (dim + 1.8)) / ((dim + 1.8 + 2 * mu_eff / dim) * (dim + 1.8))
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) * (dim + 2 + mu_eff)))

        # Initial state
        x = np.zeros(dim)
        sigma = 2.0
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        inv_C_sqrt = np.eye(dim)

        # Track best
        best_x = x.copy()
        best_f = func(x)
        calls_used = 1

        # Precompute step sizes for eigenvalue computation
        def update_inverse_sqrt(C, eps=1e-20):
            """Compute inverse square root of C using eigen decomposition."""
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, eps)
            inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            return inv_sqrt

        inv_C_sqrt = update_inverse_sqrt(C)

        for gen in range(budget // lambda_ + 1):
            # Check time
            if time.time() > deadline - 5:
                break

            # Generate population
            ys = []
            for i in range(lambda_):
                if calls_used >= budget:
                    break
                # Sample from N(0, C)
                z = rng.standard_normal(dim)
                y = x + sigma * (inv_C_sqrt @ z)
                y = np.clip(y, -5, 5)
                ys.append(y)
                calls_used += 1

            if not ys:
                break

            # Evaluate
            fs = []
            for y in ys:
                f = func(y)
                fs.append(f)
                if f < best_f:
                    best_f = f
                    best_x = y.copy()

            fs = np.array(fs)
            ys = np.array(ys)

            # Sort by fitness
            idx = np.argsort(fs)
            ys_sorted = ys[idx]
            fs_sorted = fs[idx]

            # Update mean
            y_mean = np.sum(w[:, None] * ys_sorted[:mu], axis=0)
            x = x + y_mean

            # Update sigma
            y_sigma = (ys_sorted[:mu] - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (inv_C_sqrt @ y_sigma)
            norm_ps = np.linalg.norm(p_sigma)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            sigma = max(sigma, 1e-12)

            # Update C
            y_c = (ys_sorted[:mu] - x) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
            C = (1 + c_1 * (c_mu + 1)) * C + \
                c_1 * (np.outer(p_c, p_c) - C) + \
                c_mu * np.sum(w[:, None, None] * np.outer(y_c, y_c), axis=0)

            # Symmetrize
            C = (C + C.T) / 2

            # Update inverse square root
            inv_C_sqrt = update_inverse_sqrt(C)

            # Check stagnation - if no improvement in many generations, perturb
            # (simple restart mechanism)
            # For now, just continue

        return best_x, best_f
# EVOLVE-BLOCK-END
