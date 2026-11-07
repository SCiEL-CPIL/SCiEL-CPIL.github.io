"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES with adaptive sigma, boundary handling, and deadline guard.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # 280s safety margin within 300s limit

        rng = np.random.default_rng(42)

        # CMA-ES parameters
        # Population size
        lambda_ = 4 + int(3 * np.log(dim))
        # Step size
        sigma = 2.5
        # Weights for the mean update
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        # Adaptation rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        # Covariance matrix
        C = np.eye(dim)
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        # Inverse square root of C (Cholesky based)
        # We'll use np.linalg.cholesky and solve for C^{-1/2}
        # For simplicity and robustness, we'll recompute C^{-1/2} each iteration
        # via eigendecomposition or Cholesky. Cholesky is faster for SPD.
        # C is SPD, so Cholesky works.
        # C^{-1/2} = (C^{1/2})^{-1}
        # C^{1/2} from Cholesky: C = L L^T, so C^{1/2} is not L, but we can use
        # C^{-1/2} = (L L^T)^{-1/2}. Actually, the standard CMA-ES uses
        # C^{-1/2} such that C^{-1/2} C C^{-1/2} = I.
        # We can compute C^{-1/2} via eigendecomposition: C = V D V^T,
        # C^{-1/2} = V D^{-1/2} V^T.
        # For dim=5, eigendecomposition is cheap.

        # Initial point: center of the box
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        call_count = 1

        # Precompute initial C^{-1/2}
        def compute_c_inv_sqrt(C):
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        c_inv_sqrt = compute_c_inv_sqrt(C)

        # Track best
        for gen in range(budget - 1):
            if time.time() > deadline:
                break
            if call_count >= budget:
                break

            # Sample lambda_ points
            ys = []
            fvals = []
            for i in range(lambda_):
                if time.time() > deadline:
                    break
                if call_count >= budget:
                    break
                z = rng.standard_normal(dim)
                y = x + sigma * (c_inv_sqrt @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                call_count += 1
                ys.append(y)
                fvals.append(fy)
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()

            if len(fvals) == 0:
                break

            # Sort by fitness
            fvals_arr = np.array(fvals)
            ys_arr = np.array(ys)
            idx = np.argsort(fvals_arr)
            ys_sorted = ys_arr[idx]
            fvals_sorted = fvals_arr[idx]

            # Update mean
            y_mean = (weights[:, None] * ys_sorted[:mu]).sum(axis=0)
            x_new = x + (y_mean - x)
            x_new = np.clip(x_new, -5, 5)

            # Update evolution paths
            # p_sigma = (1 - c_sigma) * p_sigma + c_sigma * sqrt(mu_eff) * C^{-1/2} @ (y_mean - x) / sigma
            diff = (y_mean - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * (c_inv_sqrt @ diff)

            # h_sigma: check if p_sigma is not too large
            chi_lambda = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) / chi_lambda if np.linalg.norm(p_sigma) < 1e10 else 0
            h_sigma = min(1.0, h_sigma)

            # p_c = (1 - c_c) * p_c + h_sigma * c_c * sqrt(mu_eff) * (y_mean - x)
            p_c = (1 - c_c) * p_c + h_sigma * c_c * np.sqrt(mu_eff) * (y_mean - x)

            # Update covariance
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c p_c^T + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * sum(weights_i * (y_i - x)(y_i - x)^T)
            outer = np.outer(p_c, p_c)
            rank1 = c_1 * (outer + (1 - h_sigma) * c_c * (2 - c_c) * C)
            rank_mu = c_mu * (weights[:, None, None] * (ys_sorted[:mu][:, None, :] - x[None, :]) * (ys_sorted[:mu][:, None, :] - x[None, :]).transpose(0, 2, 1)).sum(axis=0)
            C = (1 - c_1 - c_mu) * C + rank1 + rank_mu

            # Symmetrize
            C = (C + C.T) / 2

            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_lambda - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)

            x = x_new
            fx = fvals_sorted[0]
            if fx < best_fx:
                best_fx = fx
                best_x = ys_sorted[0].copy()

            # Recompute C^{-1/2}
            try:
                c_inv_sqrt = compute_c_inv_sqrt(C)
            except np.linalg.LinAlgError:
                # Fallback: regularize C
                C = C + 1e-10 * np.eye(dim)
                c_inv_sqrt = compute_c_inv_sqrt(C)

        return best_x, best_fx
# EVOLVE-BLOCK-END
