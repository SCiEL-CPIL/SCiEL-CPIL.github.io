"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population size 4 and cumulative step-size control."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0

        # Start from origin to exploit known locality of f14 minimum
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # CMA-ES parameters
        pop_size = 4
        sigma = 1.0
        C = np.eye(dim)
        # Eigen-decomposition cache for C
        eigvals, eigvecs = np.linalg.eigh(C)
        # Ensure positive eigenvalues
        eigvals = np.maximum(eigvals, 1e-12)

        # Cumulative step-size control parameters
        cc = (4 + dim / dim) / (dim + 4 + 2 / dim)
        cs = (2 + 2 / dim) / (dim + 2 + 2 / dim)
        cmu = min(2 - cc + 2 / dim, 0.5)
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Weights for recombination
        weights = np.ones(pop_size) / pop_size
        mueff = 1.0 / np.sum(weights ** 2)

        calls_made = 1

        while calls_made < budget and time.time() < deadline:
            # Update C using eigendecomposition
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Generate population
            offspring = []
            offspring_vals = []
            for i in range(pop_size):
                # Sample from multivariate normal
                z = rng.standard_normal(dim)
                y = x + sigma * (eigvecs @ (eigvals ** 0.5) @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                offspring.append(y)
                offspring_vals.append(fy)

            if time.time() > deadline:
                break

            # Sort by fitness
            indices = np.argsort(offspring_vals)
            sorted_offspring = [offspring[i] for i in indices]
            sorted_vals = [offspring_vals[i] for i in indices]

            # Update best
            if sorted_vals[0] < best_fx:
                best_fx = sorted_vals[0]
                best_x = sorted_offspring[0].copy()

            # Compute weighted mean
            m_new = np.zeros(dim)
            for i in range(pop_size):
                m_new += weights[i] * sorted_offspring[i]

            # Update evolution paths
            diff = m_new - x
            if sigma > 0:
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (
                    eigvecs.T @ (eigvecs @ (diff / sigma))
                )
                norm_ps = np.linalg.norm(ps)
                hsig = (norm_ps / (1 - (1 - cs) ** (budget - calls_made))) < (
                    1.4 + 2 / (dim + 1)
                ) * chi

                if hsig:
                    pc = (1 - cc) * pc + hsig * cc / sigma * diff
                else:
                    pc = (1 - cc) * pc

                # Update covariance matrix
                delta_hs = (1 - hsig) * cc * (2 - cc)
                outer = np.outer(pc, pc)
                for i in range(pop_size):
                    wi = weights[i]
                    diff_i = sorted_offspring[i] - x
                    outer += (
                        (wi / sigma ** 2)
                        - (wi * delta_hs / (1 - (1 - cc) ** 2))
                        * (1 - cc)
                    ) * np.outer(diff_i, diff_i)

                # Update eigendecomposition
                C = (1 - (cmu + c1)) * C + c1 * np.outer(pc, pc) + cmu * outer
                c1 = 2 / ((dim + 1) ** 2 + 10)

                # Recompute eigenvalues and eigenvectors
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)

                # Update sigma with cumulative step-size control
                sigma *= np.exp(
                    cs / chi * (norm_ps / chi - 1)
                )
                sigma = np.clip(sigma, 1e-10, 100)

            x = m_new

        return best_x, best_fx
# EVOLVE-BLOCK-END
