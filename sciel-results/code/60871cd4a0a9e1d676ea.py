"""Rewritten program using CMA-ES for BBOB f11 (Discus) optimization."""
# EVOLVE-BLOCK-START
import numpy as np
import math


class Optimizer:
    """CMA-ES optimizer for the Discus function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Parameters for CMA-ES
        # Population size lambda
        lam = max(4, int(math.ceil(4 + 3 * math.log(dim))))
        # Step size initialization
        sigma = 2.0
        # Initial point
        x = np.zeros(dim)
        fx = func(x)

        # Covariance matrix
        C = np.eye(dim)
        # Evolution path for covariance adaptation
        pc = np.zeros(dim)
        # Evolution path for step size adaptation
        ps = np.zeros(dim)

        # Weights
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)

        # Adaptation parameters
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Boundary handling: reflect back into box
        def clip_to_box(v):
            return np.clip(v, -5.0, 5.0)

        # Precompute Cholesky of C (initially identity)
        try:
            D = np.linalg.cholesky(C)
            B = np.linalg.inv(D).T
        except np.linalg.LinAlgError:
            D = np.eye(dim)
            B = np.eye(dim)

        calls_used = 1  # initial evaluation

        for gen in range(budget - 1):
            # Generate population
            ys = np.zeros((lam, dim))
            for i in range(lam):
                z = np.random.randn(dim)
                y = x + sigma * (B @ z)
                ys[i] = clip_to_box(y)

            # Evaluate population
            fs = np.zeros(lam)
            for i in range(lam):
                fs[i] = func(ys[i])
                calls_used += 1

            # Sort by fitness
            idx = np.argsort(fs)
            ys_sorted = ys[idx]
            fs_sorted = fs[idx]

            # Update mean
            x_old = x.copy()
            x = (weights[:, None] * ys_sorted[:mu]).sum(axis=0)

            # Update evolution paths
            ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mu_eff) / sigma * (B.T @ (x - x_old))
            hsig = (np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (2 * (gen + 1)))) < (1.4 + 2 / (dim + 1)) * math.sqrt((dim + 1) / (dim - 1))
            pc = (1 - cc) * pc + hsig * math.sqrt(cc * (2 - cc) * mu_eff) * (x - x_old) / sigma

            # Update covariance
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C = (1 + c1 * delta_hsig - c1 - cmu) * C
            C += c1 * (np.outer(pc, pc) - C)
            C += cmu * sum(wi * np.outer(ys_sorted[i] - x_old, ys_sorted[i] - x_old) for i, wi in enumerate(weights))

            # Ensure symmetry and positive semi-definite
            C = (C + C.T) / 2.0
            # Regularize if needed
            try:
                D = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
                B = np.linalg.inv(D).T
            except np.linalg.LinAlgError:
                # Reset if covariance becomes invalid
                C = np.eye(dim)
                D = np.eye(dim)
                B = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)

            # Update sigma
            sigma *= math.exp(cs / dim * (np.linalg.norm(ps) / math.sqrt(mu_eff) - 1))
            sigma = max(sigma, 1e-12)
            sigma = min(sigma, 10.0)

            # Track best
            if fs_sorted[0] < fx:
                fx = fs_sorted[0]
                x_best = ys_sorted[0].copy()
            else:
                x_best = x

            # Early stopping check
            if gen > 50:
                # Check for stagnation
                if np.all(np.abs(ys_sorted - x) < 1e-10):
                    break

        return x_best, fx
# EVOLVE-BLOCK-END
