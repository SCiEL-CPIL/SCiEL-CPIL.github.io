"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with 1/5 success rule, augmented with CMA-ES-like covariance adaptation."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        # Initialize at origin (shifts are in [-4,4]^5, so origin is a reasonable start)
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls_made = 1

        # CMA-ES style state
        sigma = 2.0
        C = np.eye(dim)  # Covariance matrix
        p_sigma = np.zeros(dim)  # Evolution path for sigma
        p_c = np.zeros(dim)      # Evolution path for C
        c_sigma = 1.0 / (4 * dim + 4)
        c_c = (2 + 2 / (dim + 1)) / (dim + 4)
        mu = dim // 2 + 1  # Number of recombination weights
        # Simple recombination weights (rank-mu)
        weights = np.ones(mu) / mu
        nu = 0.5 * (dim + 3)  # Learning rate for covariance

        for step in range(budget - 1):
            if time.time() > deadline:
                break

            # Generate mu candidates
            offspring = []
            for _ in range(mu):
                # Sample from N(x, sigma^2 * C)
                z = rng.multivariate_normal(np.zeros(dim), C)
                y = x + sigma * z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                offspring.append((fy, y))

            # Sort by fitness
            offspring.sort(key=lambda t: t[0])

            # Check if best offspring improves
            best_off_f, best_off_x = offspring[0]
            if best_off_f < fx:
                # Update evolution paths
                d = (best_off_x - x) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * d
                h_sigma = 1.0 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (step + 1))) < (1.4 + 2 / (dim + 1)) * np.trace(C) / dim else 0.0
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * d
                C = (1 - c_c * (2 - c_c) - nu) * C + c_c * (2 - c_c) * np.outer(p_c, p_c) + nu * np.sum([np.outer(offspring[i][1] - x, offspring[i][1] - x) / sigma**2 for i in range(mu)], axis=0)
                # Normalize trace
                C = C * (dim / np.trace(C))
                # Update sigma
                sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - chi_n))
                sigma = np.clip(sigma, 1e-8, 10.0)

                # Move to best offspring
                x = best_off_x
                fx = best_off_f
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                # No improvement, shrink sigma slightly
                sigma *= 0.9
                sigma = max(sigma, 1e-8)

            # Ensure C remains positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-10, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

        return best_x, best_f
# EVOLVE-BLOCK-END
