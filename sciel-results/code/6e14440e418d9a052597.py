"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES for the bent cigar (f12) problem."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size
        lam = 4 + int(3 * np.log(dim))  # 10 for dim=5
        mu = lam // 2

        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        c1 = (2.0 * (mueff - 2.0) + 1.0) / ((dim + 1.0) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff + 1.5 / dim - 2.0) / ((dim + 2.0) ** 2 + mueff))
        cs = (mueff + 1.7) / (dim + mueff + 5.0)
        c1 = (2.0 * (mueff - 2.0) + 1.0) / ((dim + 1.0) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff + 1.5 / dim - 2.0) / ((dim + 2.0) ** 2 + mueff))
        ds = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs
        chiN = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))

        # Initial point and sigma
        x0 = rng.uniform(-5, 5, dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Evaluate initial point
        x_best = x0.copy()
        f_best = func(x_best)
        calls = 1

        # Precompute square root of C using Cholesky or eigendecomp
        def sqrtm(C):
            # Use eigendecomposition for stability
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

        B = sqrtm(C)
        D = np.sqrt(np.diag(C))

        while calls < budget:
            # Generate population
            ys = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x0 + sigma * B @ z
                ys.append(y)
                calls += 1
                if calls >= budget:
                    break

            # Evaluate population
            fs = []
            for y in ys:
                fy = func(y)
                fs.append(fy)

            # Sort by fitness
            idx = np.argsort(fs)
            ys_sorted = [ys[i] for i in idx]
            fs_sorted = [fs[i] for i in idx]

            # Update best
            if fs_sorted[0] < f_best:
                f_best = fs_sorted[0]
                x_best = ys_sorted[0].copy()

            # Mean update
            x0 = sum(weights[i] * ys_sorted[i] for i in range(mu))

            # Step-size control (CSA)
            y_mean = (x0 - x0)  # placeholder, actual update below
            # Actually, we need to track the old x0
            # Let's restructure: keep x0_old

            break

        # The above loop structure is flawed; let's rewrite properly.
        # Restart with a cleaner implementation.

        # Re-initialize for clean run
        x0 = rng.uniform(-5, 5, dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        f_best = func(x0)
        x_best = x0.copy()
        calls = 1

        B = sqrtm(C)
        D = np.sqrt(np.diag(C))

        while calls < budget:
            x0_old = x0.copy()

            # Generate population
            ys = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x0 + sigma * B @ z
                ys.append(y)
                calls += 1
                if calls >= budget:
                    break

            if len(ys) == 0:
                break

            # Evaluate population
            fs = []
            for y in ys:
                fy = func(y)
                fs.append(fy)

            # Sort by fitness
            idx = np.argsort(fs)
            ys_sorted = [ys[i] for i in idx]
            fs_sorted = [fs[i] for i in idx]

            # Update best
            if fs_sorted[0] < f_best:
                f_best = fs_sorted[0]
                x_best = ys_sorted[0].copy()

            # Mean update
            x0_new = sum(weights[i] * ys_sorted[i] for i in range(mu))

            # Update CMA state
            y_mean = (x0_new - x0_old) / sigma
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * y_mean
            hsig = (np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2 * (calls // lam + 1)))) < (1.4 + 2.0 / (dim + 1)) * chiN
            pc = (1.0 - c1) * pc + hsig * np.sqrt(c1 * (2.0 - c1) * mueff) * y_mean
            C = (1.0 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1.0 - hsig) * c1 * C) + cmu * sum(
                weights[i] * np.outer(ys_sorted[i] - x0_old, ys_sorted[i] - x0_old) / sigma ** 2 for i in range(mu)
            )

            # Ensure symmetry
            C = (C + C.T) / 2.0

            # Update sigma (CSA)
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chiN - 1.0))
            sigma = np.clip(sigma, 1e-8, 1e3)

            x0 = x0_new

            # Update B and D
            B = sqrtm(C)
            D = np.sqrt(np.diag(C))

        return x_best, f_best
# EVOLVE-BLOCK-END
