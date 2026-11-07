"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-lite for 5D BBOB f6 with bounded box, time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        rng = np.random.default_rng(42)

        dim = self.dim
        box_lo, box_hi = -5.0, 5.0

        # ---- CMA-ES parameters ----
        # Population size
        N = max(4, int(4 + 3 * np.log(dim)))
        # Strategy parameters
        c_sigma = (2 + 2.0 / (dim + 1)) / ((dim + 1.3) + 2 + 2.0 / (dim + 1))
        d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((N - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + N / dim) / (dim + 4 + 2 * N / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + N)
        c_mu = min(
            1 - c_1,
            2 * ((N / 2 + 1) / (dim + 2) - 1) / ((dim + 2) ** 2 + N),
        )
        chi_N = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial step size
        sigma = 2.0
        # Initial mean in the center of the box
        x_mean = np.zeros(dim)

        # Covariance matrix
        C = np.eye(dim)
        # Inverse square root of C (maintained via eigendecomp each step for simplicity)
        C_eigvals = np.ones(dim)
        C_eigvecs = np.eye(dim)

        # Cumulation paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        # h_sigma: expectation of ||z|| for chi_N
        h_sigma = (c_sigma + 1) / (c_sigma + d_sigma) * (chi_N - 1)

        # Weights for recombination
        mu = N // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()

        calls_made = 0
        x_best = x_mean.copy()
        f_best = float("inf")

        def _safe_eig(C_mat):
            """Eigendecomposition with numerical safeguards."""
            C_sym = 0.5 * (C_mat + C_mat.T)
            try:
                evals, evecs = np.linalg.eigh(C_sym)
            except np.linalg.LinAlgError:
                evals = np.ones(dim)
                evecs = np.eye(dim)
            evals = np.maximum(evals, 1e-12)
            return evals, evecs

        def _make_step_sigma(evals, evecs):
            """Compute C^{1/2} for sampling: evecs @ diag(sqrt(evals)) @ evecs.T applied to z."""
            # We'll compute the matrix M = evecs @ diag(1/sqrt(evals)) @ evecs.T
            # Then y = x + sigma * M @ z  where z ~ N(0, I)
            inv_sqrt = 1.0 / np.sqrt(evals)
            # M = evecs @ diag(inv_sqrt) @ evecs.T
            # We can precompute the product
            M = (evecs * inv_sqrt) @ evecs.T
            return M

        while calls_made < self.budget:
            # Time check
            if time.time() > deadline:
                break

            # ---- Sample N points ----
            evals, evecs = _safe_eig(C)
            M = _make_step_sigma(evals, evecs)

            Y = np.zeros((N, dim))
            Z = np.zeros((N, dim))
            F = np.zeros(N)

            for i in range(N):
                if time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                Z[i] = z
                y_raw = x_mean + sigma * (M @ z)
                y = np.clip(y_raw, box_lo, box_hi)
                Y[i] = y
                F[i] = func(y)
                calls_made += 1
                if F[i] < f_best:
                    f_best = F[i]
                    x_best = y.copy()

            if calls_made < N:
                break

            # ---- Sort by fitness ----
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]
            Z_sorted = Z[order]

            # Update best
            if F_sorted[0] < f_best:
                f_best = F_sorted[0]
                x_best = Y_sorted[0].copy()

            # ---- Weighted mean ----
            x_new_mean = np.zeros(dim)
            for i in range(mu):
                x_new_mean += weights[i] * Y_sorted[i]

            # ---- Cumulation path for sigma ----
            # Need C^{-1/2} @ (x_new_mean - x_mean) / sigma
            inv_sqrt_evals = 1.0 / np.sqrt(evals)
            # C^{-1/2} = evecs @ diag(inv_sqrt) @ evecs.T
            C_inv_sqrt = (evecs * inv_sqrt_evals) @ evecs.T
            delta = (x_new_mean - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ delta)

            # h_sigma check
            if np.linalg.norm(p_sigma) < (1.4 + 2 / (dim + 1)) * h_sigma * np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made + 1))):
                pass  # normal
            # Note: the exact h_sigma threshold uses the ratio of expectations; we use a simplified check

            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_N - 1))
            # Clamp sigma
            sigma = np.clip(sigma, 1e-10, 50.0)

            # ---- Cumulation path for C ----
            # C^{-1/2} @ (x_new_mean - x_mean) / sigma  = C_inv_sqrt @ delta  (already computed)
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv_sqrt @ delta)

            # Update C
            rank_one = np.outer(p_c, p_c)
            h_c = (1 - c_c) ** 2
            # The correction term
            C += c_1 * (
                (1 + (2 * h_c - 1)) * np.outer(C_inv_sqrt @ delta, C_inv_sqrt @ delta)
                - np.eye(dim)
            )
            for i in range(mu):
                C += c_mu * weights[i] * np.outer(
                    C_inv_sqrt @ (Y_sorted[i] - x_mean) / sigma,
                    C_inv_sqrt @ (Y_sorted[i] - x_mean) / sigma,
                )

            # Symmetrize
            C = 0.5 * (C + C.T)

            x_mean = x_new_mean

            # Stop if sigma is tiny
            if sigma < 1e-10:
                break

        return x_best, f_best
# EVOLVE-BLOCK-END
