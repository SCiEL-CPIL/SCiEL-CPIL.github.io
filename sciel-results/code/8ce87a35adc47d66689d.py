"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Global random sampling + CMA-ES optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Phase 1: Global random sampling to find a good initial point
        n_global = 30
        if budget <= n_global:
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            for _ in range(budget - 1):
                y = rng.uniform(-5, 5, dim)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
            return x, fx

        best_x = None
        best_f = np.inf
        for _ in range(n_global):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES from the best global point
        remaining = budget - n_global
        x = best_x.copy()
        fx = best_f

        # CMA-ES parameters - tuned for 5D
        popsize = 9  # 4 + 3*ln(6) ≈ 9
        popsize = min(popsize, max(2, remaining // 3))

        sigma = 2.0  # Larger initial step to explore more
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        cs = (popsize + 2) / (dim + 4)
        ds = 1 + 2 * max(0, (np.sqrt((popsize + 1) / (dim + 3)) - 1) / (dim + 1)) + cs
        c1 = (2 + 2.0 / (dim + 1.3)) / ((dim + 2) ** 2 + popsize)
        cmu = min(1 - c1, 2 * (popsize - 2 + 1.0 / popsize) / ((dim + 2) ** 2 + popsize))
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Weights
        w = np.exp(-5 * np.arange(popsize) / popsize)
        w /= w.sum()
        mu_eff = (w ** 2).sum() ** -1

        global_best_x = x.copy()
        global_best_f = fx

        calls_left = remaining
        while calls_left >= popsize:
            # Generate population
            z = rng.standard_normal((popsize, dim))
            # Transform using Cholesky of C
            L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            candidates = np.array([
                np.clip(x + sigma * (L @ z[i]), -5, 5) for i in range(popsize)
            ])

            # Evaluate
            fvals = np.array([func(c) for c in candidates])
            calls_left -= popsize

            # Update global best
            idx_best = np.argmin(fvals)
            if fvals[idx_best] < global_best_f:
                global_best_f = fvals[idx_best]
                global_best_x = candidates[idx_best].copy()

            # Sort
            order = np.argsort(fvals)
            fvals_sorted = fvals[order]
            x_sorted = candidates[order]

            # Save old mean
            x_old = x.copy()

            # Update mean
            x = np.sum(w[:, None] * x_sorted, axis=0)

            # Update ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / chi * (L.T @ (x - x_old))

            # Update pc
            pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mu_eff) * (x - x_old) / sigma

            # Update C - correct CMA-ES update
            delta_hs = (np.sum(ps ** 2) / (1 - (1 - cs) ** 2)) / (dim * (1 - (1 - cs) ** 2)) - 1
            hsig = 1.0 if delta_hs < (1 + 4 / (dim + 1)) else 0.0

            # Standard CMA-ES covariance update
            C = (1 + c1 * hsig + cmu - c1 - cmu) * C
            C += c1 * (np.outer(pc, pc) - (1 - hsig) * C)
            C += cmu * np.sum(w[:, None, None] * np.outer(x_sorted - x_old, x_sorted - x_old), axis=0) / sigma ** 2

            # Ensure positive semi-definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi - 1))
            sigma = np.clip(sigma, 1e-5, 10.0)

            # Update local best
            if fvals_sorted[0] < fx:
                fx = fvals_sorted[0]
                x = x_sorted[0].copy()

        # Use remaining calls with smaller population or single steps
        while calls_left > 0:
            n_gen = min(popsize, calls_left)
            z = rng.standard_normal((n_gen, dim))
            L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            candidates = np.array([
                np.clip(x + sigma * (L @ z[i]), -5, 5) for i in range(n_gen)
            ])
            fvals = np.array([func(c) for c in candidates])
            calls_left -= n_gen

            idx_best = np.argmin(fvals)
            if fvals[idx_best] < global_best_f:
                global_best_f = fvals[idx_best]
                global_best_x = candidates[idx_best].copy()

            if fvals[idx_best] < fx:
                fx = fvals[idx_best]
                x = candidates[idx_best].copy()

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
