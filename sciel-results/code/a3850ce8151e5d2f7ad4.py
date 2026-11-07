"""CMA-ES optimizer for BBOB f19 (Griewank-Rosenbrock)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive step size for BBOB f19."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        N = max(4, int(4 + 3 * np.log(dim)))  # population size
        # Initial center and sigma
        x_mean = np.zeros(dim)
        sigma = 2.0

        # Covariance matrix
        C = np.eye(dim)
        # Inverse square root of C (for step size adaptation)
        # We'll use a simple approach: track the matrix and update it
        C_inv_sqrt = np.eye(dim)

        # Step size adaptation parameters
        cs = 1 / (N + 4)
        ds = 1 + 2 * max(0, (N + 1) / (dim + 2) - 1) + cs
        # Path length parameters
        c_sigma = (N + 2) / (N + 4 + 2 * dim / (dim + 2))
        c_c = (4 + N / dim) / (N + 4 + 3 * dim / (dim + 2))

        # Update paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Weighting
        w = 0.5 * np.ones(N)
        w /= np.sum(w)
        mu = N // 2
        weights = np.zeros(N)
        weights[:mu] = w[:mu]
        mu_eff = 1.0 / np.sum(weights[:mu] ** 2)

        # Track best
        x_best = x_mean.copy()
        f_best = func(x_best)
        call_count = 1

        # Evaluation counter
        def safe_call(x):
            nonlocal call_count
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            call_count += 1
            return x_clipped, fx

        for iteration in range(budget // N + 1):
            # Check deadline
            if time.time() > deadline - 5:
                break
            if call_count >= budget:
                break

            # Generate offspring
            x_new = np.zeros((N, dim))
            f_new = np.zeros(N)

            for i in range(N):
                if call_count >= budget or time.time() > deadline - 2:
                    # Use the best so far
                    x_new[i] = x_best
                    f_new[i] = f_best
                    continue

                # Generate step
                z = rng.standard_normal(dim)
                # Apply C_inv_sqrt (approximated by identity for simplicity in this version)
                # For a proper CMA-ES, we'd update C and its inverse sqrt
                step = sigma * C_inv_sqrt @ z
                x_candidate = x_mean + step
                x_clipped, f_val = safe_call(x_candidate)
                x_new[i] = x_clipped
                f_new[i] = f_val

            # Sort by fitness
            order = np.argsort(f_new)
            x_new = x_new[order]
            f_new = f_new[order]

            # Update best
            if f_new[0] < f_best:
                f_best = f_new[0]
                x_best = x_new[0].copy()

            # Update mean
            x_old = x_mean.copy()
            x_mean = np.sum(weights[:, None] * x_new[:mu], axis=0)

            # Update step size
            # Compute new p_sigma
            # Need C_inv_sqrt @ (x_mean - x_old) / sigma
            diff = (x_mean - x_old) / sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu_eff) * C_inv_sqrt @ diff

            # Update sigma
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 100)

            # Update covariance matrix (simplified: use rank-one and rank-mu updates)
            # For simplicity, we'll do a basic rank-one update
            y = (x_mean - x_old) / sigma
            C = (1 + c_c * (2 - c_c) * mu_eff) * C - c_c * np.outer(y, y)
            # Ensure positive semi-definite
            # Add small regularization
            C += 1e-12 * np.eye(dim)

            # Update C_inv_sqrt (approximate)
            # For a proper implementation, we'd use eigendecomposition
            # Here we'll just use identity as a simplification
            C_inv_sqrt = np.eye(dim)

        return x_best, f_best
# EVOLVE-BLOCK-END
