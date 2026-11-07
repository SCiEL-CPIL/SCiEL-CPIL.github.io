"""Optimized solver for BBOB f12 (Bent Cigar) using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for the Bent Cigar function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        lam = int(4 + 3 * np.log(dim))  # ~9 for dim=5
        mu = lam // 2  # number of parents
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # Initial point and sigma
        x = np.zeros(dim)
        sigma = 2.0
        fx = func(x)

        # Best tracking
        best_x = x.copy()
        best_f = fx
        calls = 1

        # Covariance matrix and evolution paths
        C = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Parameters
        cs = (mueff + 2) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        c1 = (2 / ((dim + 1.3) ** 2 + mueff)) * (0.5 + 2 / (dim + mueff))
        cmu = min(1 - c1, 2 * ((mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff)))
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs

        while calls < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate offspring
            ys = []
            fs = []
            for i in range(lam):
                if time.time() > deadline:
                    break
                # Sample from N(0, C) using Cholesky
                L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                y = L @ rng.standard_normal(dim)
                x_new = np.clip(x + sigma * y, -5, 5)
                f_new = func(x_new)
                calls += 1
                ys.append(y)
                fs.append(f_new)

                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()

            if len(ys) == 0:
                break

            ys = np.array(ys)
            fs = np.array(fs)

            # Sort by fitness
            idx = np.argsort(fs)
            ys_sorted = ys[idx]
            fs_sorted = fs[idx]

            # Update best
            if fs_sorted[0] < best_f:
                best_f = fs_sorted[0]
                best_x = np.clip(x + sigma * ys_sorted[0], -5, 5).copy()

            # Mean update
            x_old = x.copy()
            x = x_old + sigma * np.sum(weights[:, None] * ys_sorted[:mu], axis=0)

            # Evolution paths
            # Compute C^{-1/2}
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            # Update ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * C_inv_sqrt @ (x - x_old) / sigma

            # Update pc
            if np.linalg.norm(ps) / chiN < (1.4 + cs) * np.sqrt((mueff + 1) / (dim + 1)):
                pc = (1 - cs) * pc + cs * C_inv_sqrt @ (x - x_old) / sigma

            # Step size adaptation
            delta_sig = np.linalg.norm(ps) / chiN - (1 - cs ** (2 * (budget - calls)))
            delta_sig = np.clip(delta_sig, -5, 5)
            sigma *= np.exp((1 / damps) * delta_sig)
            sigma = np.clip(sigma, 1e-10, 100)

            # Covariance matrix update
            outer = np.outer(x - x_old, x - x_old)
            C = (1 + c1 + cmu - c1) * C + c1 * (np.outer(pc, pc) - (2 * cs - cs ** 2) * C) \
                + cmu * np.sum(weights[:, None, None] * np.outer(ys_sorted[:mu], ys_sorted[:mu]), axis=0) / sigma ** 2

            # Ensure symmetry and positive semi-definite
            C = 0.5 * (C + C.T)
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.min(eigvals) < 1e-10:
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

        return best_x, best_f
# EVOLVE-BLOCK-END
