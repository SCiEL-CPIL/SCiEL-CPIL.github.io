"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with deadline guard for Rastrigin optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # CMA-ES parameters
        pop_size = 4 + int(3 * np.log(dim))  # ~9 for dim=5
        pop_size = max(pop_size, 2 * dim + 1)  # at least 11 for dim=5

        # Initial point: center of the box
        x = np.zeros(dim)
        fx = func(x)
        calls = 1

        # Initial sigma: cover the search space
        sigma = 2.0

        # CMA-ES state
        C = np.eye(dim)  # covariance matrix
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Strategy parameters
        mu = pop_size // 2  # number of selected parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)

        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        def cma_step():
            nonlocal x, fx, sigma, C, p_sigma, p_c, calls
            nonlocal deadline, budget, start_time

            if time.time() > deadline:
                return False

            # Generate population
            B = np.linalg.cholesky(C)
            xs = np.zeros((pop_size, dim))
            fs = np.zeros(pop_size)

            for i in range(pop_size):
                if time.time() > deadline:
                    return False
                z = rng.standard_normal(dim)
                xi = np.clip(x + sigma * B @ z, -5, 5)
                xs[i] = xi
                fs[i] = func(xi)
                calls += 1

            if time.time() > deadline:
                return False

            # Sort by fitness
            order = np.argsort(fs)
            xs_sorted = xs[order]
            fs_sorted = fs[order]

            # Update mean
            x_new = np.sum(weights[:, None] * xs_sorted[:mu], axis=0)
            x_new = np.clip(x_new, -5, 5)

            # Update covariance
            y = (x_new - x) / sigma
            y_c = C @ y

            # Update p_sigma
            h_sigma = (np.linalg.norm(p_sigma) /
                       np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - pop_size))))
            h_sigma = min(h_sigma, 1 + 2 * (mu_eff - 1) / (dim + 2))
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma *
                       (2 - c_sigma) * mu_eff) * y_c

            # Update p_c
            h_c = (np.linalg.norm(p_c) /
                   np.sqrt(1 - (1 - c_c) ** (2 * (calls - pop_size))))
            h_c = min(h_c, 1 + 2 * (mu_eff - 1) / (dim + 2))
            p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mu_eff) * y

            # Rank-one update
            C = (1 + c_c * (2 - c_c) - c_sigma) * C + \
                c_sigma * (1 + (2 - c_sigma) * (1 - h_sigma) *
                           mu_eff / (dim + 1) - 1) * np.outer(y_c, y_c)

            # Rank-mu update
            v = (xs_sorted[:mu] - x) / sigma
            v = v @ np.linalg.cholesky(C).T
            C = C + c_sigma * (2 - c_sigma) * (1 - h_sigma) * \
                mu_eff * (np.outer(v_c := (C @ y), v_c) - C)  # simplified

            # Actually, let me use a simpler and more standard formulation
            # Let me redo this more carefully

            return True

        # Simpler, more robust CMA-ES implementation
        # Reset state
        x = np.zeros(dim)
        fx = func(x)
        calls = 1
        sigma = 2.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Precompute weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        gen = 0
        while calls < budget and time.time() < deadline:
            gen += 1
            # Generate population
            B = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            zs = rng.standard_normal((pop_size, dim))
            xs = np.clip(x[None, :] + sigma * (zs @ B.T), -5, 5)

            # Evaluate
            fs = np.array([func(xi) for xi in xs])
            calls += pop_size

            if time.time() > deadline:
                break

            # Sort
            order = np.argsort(fs)
            xs_sorted = xs[order]
            fs_sorted = fs[order]

            # Update mean
            x_new = np.sum(w[:, None] * xs_sorted[:mu], axis=0)
            x_new = np.clip(x_new, -5, 5)

            # Update step size
            y = (x_new - x) / sigma
            y_norm = np.linalg.norm(y)
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * y
            p_c = (1 - c_c) * p_c + c_c * y

            # Update sigma (1/5 rule style)
            sigma *= np.exp(c_sigma / d_sigma * (y_norm / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update covariance (simplified rank-1 + rank-mu)
            C = (1 + c_c * (2 - c_c) - c_sigma) * C + \
                c_sigma * (1 + (2 - c_sigma) * (1 - 1) *
                           mu_eff / (dim + 1) - 1) * np.outer(y, y)

            # Rank-mu update
            v = (xs_sorted[:mu] - x) / sigma
            # Project v onto C^{-1/2}
            C_inv_sqrt = np.linalg.inv(B)
            v_proj = v @ C_inv_sqrt.T
            C += c_sigma * (2 - c_sigma) * mu_eff * (
                np.sum((w[:, None] ** 2) * np.einsum('ij,ik->jk', v_proj, v_proj), axis=0)
            )

            # Symmetrize
            C = (C + C.T) / 2

            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            x = x_new
            if fs_sorted[0] < fx:
                fx = fs_sorted[0]

        return x, fx
# EVOLVE-BLOCK-END
