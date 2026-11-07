"""Rewritten optimizer using CMA-ES with adaptive population and sigma control."""
import numpy as np


class Optimizer:
    """CMA-ES style continuous optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        # Initial point
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # CMA-ES parameters
        # Population size
        lam = max(4, int(4 + 3 * np.log(dim)))  # ~4+3*log(5)=~9
        mu = lam // 2  # number of parents
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        # Step size
        sigma = 2.0
        # Covariance matrix
        C = np.eye(dim)
        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        # Learning rates
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        # Damping for sigma
        dsigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        # chi_n expected value of norm of standard normal
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        calls_made = 1
        remaining = budget - 1

        while remaining > 0:
            # Generate population
            offspring = []
            for i in range(lam):
                # Sample from multivariate normal with covariance C
                # Use Cholesky decomposition
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                z = rng.standard_normal(dim)
                y = x + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                remaining -= 1
                offspring.append((y, fy))
                if remaining <= 0:
                    break

            if not offspring:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[1])

            # Select parents
            parents = offspring[:mu]
            # Update mean
            new_x = sum(w * p[0] for w, p in zip(weights, parents))
            new_x = np.clip(new_x, -5, 5)

            # Update evolution paths
            y_mean = (new_x - x) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y_mean
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made - 1)))) < (1.4 + 2 / (dim + 1)) * chi_n
            pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mueff) * y_mean

            # Update covariance
            delta = 0 if hsig else 1
            C = (1 + c1 * delta + cmu - c1 - cmu) * C
            C += c1 * (np.outer(pc, pc) - C)
            for w, p in zip(weights, parents):
                yk = (p[0] - x) / sigma
                C += cmu * w * np.outer(yk, yk)

            # Update sigma
            sigma *= np.exp((1 / dsigma) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100)

            # Update best
            if parents[0][1] < best_fx:
                best_fx = parents[0][1]
                best_x = parents[0][0].copy()

            x = new_x
            fx = best_fx  # Not strictly used, but keep track

            # Ensure C is positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

        return best_x, best_fx
