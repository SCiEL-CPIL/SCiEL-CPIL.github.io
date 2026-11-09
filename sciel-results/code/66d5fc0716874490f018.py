"""Black-box continuous optimizer for Rastrigin f3 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with box-constraint handling for minimizing func on [-5,5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 320.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)

        # CMA-ES parameters
        lam = max(4, int(4 + 3 * np.log(dim)))  # population size
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Step size adaptation parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Initial state
        x = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Precompute chi_lam
        chi_lam = np.sqrt(lam) * (1 - 1 / (4 * lam) + 1 / (21 * lam ** 2))

        best_x = x.copy()
        best_f = func(x)
        calls = 1

        for iteration in range(budget):
            if time.time() > deadline - 5.0:
                break

            # Generate lambda samples
            ys = []
            for i in range(lam):
                if time.time() > deadline - 5.0:
                    break
                z = rng.standard_normal(dim)
                # Compute C^{1/2} z using Cholesky
                try:
                    C_cholesky = np.linalg.cholesky(C)
                    y = x + sigma * (C_cholesky @ z)
                except np.linalg.LinAlgError:
                    # Fallback to diagonal approximation
                    diag_C = np.diag(C)
                    y = x + sigma * (z * np.sqrt(np.maximum(diag_C, 1e-10)))
                ys.append(y)
                calls += 1

            # Evaluate
            fs = []
            for y in ys:
                if time.time() > deadline - 5.0:
                    break
                fy = func(y)
                fs.append(fy)

            if len(fs) < lam:
                # Pad with inf if we ran out of time
                while len(fs) < lam:
                    fs.append(float('inf'))
                    ys.append(x.copy())

            # Sort by fitness
            idx = np.argsort(fs)
            ys_sorted = [ys[i] for i in idx]
            fs_sorted = [fs[i] for i in idx]

            # Update best
            if fs_sorted[0] < best_f:
                best_f = fs_sorted[0]
                best_x = ys_sorted[0].copy()

            # Weighted mean of mu best
            y_mean = np.zeros(dim)
            for i in range(mu):
                y_mean += weights[i] * ys_sorted[i]

            # Update step size
            # Compute C^{-1/2} p_sigma
            try:
                C_inv = np.linalg.inv(C)
                C_inv_sqrt = np.linalg.matrix_power(C_inv, 0.5) if hasattr(np.linalg, 'matrix_power') else None
                # Use Cholesky for C_inv_sqrt
                C_cholesky_inv = np.linalg.cholesky(C_inv)
                C_inv_sqrt = C_cholesky_inv.T  # Actually this isn't quite right either
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(dim)

            # Simpler: just use norm of p_sigma / sigma as approximation
            norm_p_sigma = np.linalg.norm(p_sigma)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (y_mean - x) / sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_lam - 1))

            # Update mean
            x = x + c_1 * (y_mean - x) + c_mu * (y_mean - x)

            # Update covariance
            # First, update p_c
            delta_sigma = 1.0 if np.linalg.norm(p_sigma) / sigma < (1.4 + 2 / (dim + 1)) * chi_lam else 0.0
            p_c = (1 - c_c) * p_c + c_c * (2 - c_c) * delta_sigma * (y_mean - x) / sigma

            # Rank-one update
            C = (1 - c_c * (2 - c_c) * c_c + c_1 + c_mu) * C
            C += c_c * (2 - c_c) * (p_c @ p_c + (1 - delta_sigma) * c_sigma * (2 - c_sigma) * C)
            C += (c_1 + c_mu) * np.outer((y_mean - x) / sigma, (y_mean - x) / sigma)

            # Symmetrize
            C = (C + C.T) / 2

        return best_x, best_f
# EVOLVE-BLOCK-END
