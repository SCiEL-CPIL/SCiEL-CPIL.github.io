"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 280.0

        n = self.dim
        x = rng.uniform(-5, 5, n)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        sigma = 2.0
        calls_made = 1
        remaining = self.budget - 1

        # 1/5 success rule parameters
        s = 0.0
        window = 15
        successes = 0

        # Covariance matrix for rotation-invariance
        C = np.eye(n)
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n * n))

        while remaining > 0:
            if time.time() > deadline:
                break

            # Generate a single trial point
            z = rng.standard_normal(n)
            y = x + sigma * z
            y = np.clip(y, -5, 5)
            fy = func(y)
            calls_made += 1
            remaining -= 1

            success = fy < fx
            if success:
                x = y.copy()
                fx = fy
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                successes += 1

            # Update sigma using 1/5 success rule
            s = (1 - 1.0 / window) * s + (1.0 / window) * success
            sigma *= np.exp(0.33 * (s - 1.0 / 5.0))
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Update covariance matrix (simplified CMA-like update)
            if success:
                # Rank-one update in the direction of improvement
                step = (y - (y - sigma * z)) / sigma  # = z
                C = 0.99 * C + 0.01 * np.outer(z, z)
            else:
                # Decay back toward identity
                C = 0.995 * C + 0.005 * np.eye(n)

            # Symmetrize
            C = (C + C.T) / 2

            # Periodic eigenvalue repair
            if calls_made % 50 == 0:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

        return best_x, best_f
# EVOLVE-BLOCK-END
