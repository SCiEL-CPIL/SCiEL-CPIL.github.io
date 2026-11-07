"""Rewritten optimizer: CMA-ES with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with adaptive step size and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s kill

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(dim)))  # population size
        mu = lambda_ // 2  # number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initialize
        rng = np.random.default_rng(42)
        x = rng.uniform(-4, 4, dim)  # start near center
        fx = func(x)

        sigma = 2.0
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        step_size = np.ones(dim)

        best_x = x.copy()
        best_fx = fx
        calls_made = 1

        # Precompute
        def cholesky_decompose(A):
            """Compute Cholesky decomposition of A (must be SPD)."""
            L = np.linalg.cholesky(A)
            return L

        def update_sqrt(C, C_sqrt):
            """Update square root of C."""
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback: eigenvalue decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
            return C_sqrt

        def apply_sqrt(C_sqrt, v):
            """Apply C_sqrt to vector v."""
            return C_sqrt @ v

        def apply_inv_sqrt(C_inv_sqrt, v):
            """Apply inverse of C_sqrt to vector v."""
            return C_inv_sqrt @ v

        def update_inv_sqrt(C_sqrt):
            """Compute inverse of C_sqrt."""
            try:
                C_inv_sqrt = np.linalg.inv(C_sqrt)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C_sqrt @ C_sqrt.T)
                eigvals = np.clip(eigvals, 1e-10, None)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            return C_inv_sqrt

        C_inv_sqrt = update_inv_sqrt(C_sqrt)

        # Main loop
        for gen in range(1, budget + 1):
            if time.time() > deadline:
                break

            # Generate population
            ys = []
            fvals = []
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                y = x + sigma * apply_sqrt(C_sqrt, z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                ys.append(y)
                fvals.append(fy)
                calls_made += 1

            # Sort by fitness
            order = np.argsort(fvals)
            ys_sorted = [ys[i] for i in order]
            fvals_sorted = [fvals[i] for i in order]

            # Update best
            if fvals_sorted[0] < best_fx:
                best_fx = fvals_sorted[0]
                best_x = ys_sorted[0].copy()

            # Compute mean of best mu points
            y_mean = sum(weights[i] * ys_sorted[i] for i in range(mu))

            # Update evolution paths
            delta_x = (y_mean - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * delta_x
            p_c = (1 - c_c) * p_c + c_c * delta_x

            # Update covariance matrix
            delta_hsig = (
                np.sqrt(
                    c_sigma * (2 - c_sigma) * mueff * (1 - (1 - c_sigma) ** (2 * gen))
                )
                * np.linalg.norm(p_sigma)
                / chi
                - (1 - c_sigma)
            )
            C = (
                (1 + c_sigma * delta_hsig * (c_sigma + 2) - c_1 - c_mu) * C
                + c_1 * (np.outer(p_c, p_c) + (c_c - c_c * (1 - c_c)) * np.eye(dim))
                + c_mu * sum(weights[i] * np.outer(ys_sorted[i] - x, ys_sorted[i] - x) for i in range(mu))
            )

            # Ensure symmetry
            C = (C + C.T) / 2

            # Update step size
            sigma *= np.exp(
                c_sigma / chi * (np.linalg.norm(p_sigma) / chi - 1)
            )
            sigma = np.clip(sigma, 1e-10, 1e5)

            # Update x
            x = y_mean

            # Update square root and inverse square root every few generations
            if gen % 5 == 0 or gen == 1:
                C_sqrt = update_sqrt(C, C_sqrt)
                C_inv_sqrt = update_inv_sqrt(C_sqrt)

        return best_x, best_fx
# EVOLVE-BLOCK-END
