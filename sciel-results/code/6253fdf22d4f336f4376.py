"""Rewritten optimizer: Halton space-filling sweep + single tuned CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def _halton(dim, n, base=2):
    """Deterministic Halton sequence in [0,1)^dim, well-spread across the box."""
    pts = np.zeros((n, dim))
    for d in range(dim):
        b = base + d
        f = np.ones(n, dtype=np.float64)
        i = np.arange(1, n + 1, dtype=np.float64)
        while np.any(f > 0.0):
            i = i // b
            f = f / b
        pts[:, d] = f
    return pts


class Optimizer:
    """Halton sweep for early coverage, then a single CMA-ES run on the best point."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        calls_used = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def counted_func(x):
            nonlocal calls_used, best_x, best_f
            if calls_used >= budget:
                return 1e10
            x_clipped = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            calls_used += 1
            fx = func(x_clipped)
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # ---- Phase 1: deterministic multi-scale Halton sweep ----
        # Spread across the full box [-5,5]^dim for broad early coverage.
        n_halton = min(120, max(20, budget // 4))
        h = _halton(dim, n_halton)
        h = -5.0 + 10.0 * h  # map to [-5,5]^dim
        # Add a few extra structured anchor points for very early signal.
        anchors = np.array([
            np.zeros(dim),
            np.ones(dim) * 1.0,
            -np.ones(dim) * 1.0,
            np.ones(dim) * 2.0,
            -np.ones(dim) * 2.0,
        ])
        for p in np.vstack([anchors, h]):
            if calls_used >= budget or time.time() > deadline:
                break
            counted_func(p)

        # ---- Phase 2: single tuned CMA-ES from the best Halton point ----
        if calls_used < budget and time.time() < deadline:
            x0 = best_x.copy()
            # Initial sigma: scale to box size (minimizer known to lie in [-4,4]).
            sigma = 1.5

            N = dim + int(round(3 * np.log(dim + 1)))  # population size
            N = max(N, dim + 1)
            N = min(N, max(2, (budget - calls_used) // (dim + 1)))

            # CMA-ES weights (mu-weighted recombination)
            mu = N // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1.0 / np.sum(w ** 2)
            c_sigma = (mueff + 2) / (dim + mueff + 5)
            d_sigma = 1.0 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
            c_1 = (2 - mueff / (dim + 1.7) + c_sigma) / ((dim + 1.5) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
            chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            C = np.eye(dim)
            # Cholesky factor of C (start with identity)
            C_sqrt = np.eye(dim)
            C_inv_sqrt = np.eye(dim)

            def recompute_chol():
                nonlocal C_sqrt, C_inv_sqrt
                C_sym = 0.5 * (C + C.T)
                try:
                    L = np.linalg.cholesky(C_sym + 1e-12 * np.eye(dim))
                    C_sqrt = L.T  # such that C_sqrt @ C_sqrt.T == C
                except np.linalg.LinAlgError:
                    C_sqrt = np.eye(dim)
                # C^{-1/2} via eigendecomposition for stable inverse square root
                eigvals, eigvecs = np.linalg.eigh(C_sym)
                eigvals = np.clip(eigvals, 1e-12, None)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                return C_sqrt, C_inv_sqrt

            recompute_chol()

            max_gen = (budget - calls_used) // N
            for gen in range(max_gen):
                if time.time() > deadline or calls_used >= budget:
                    break

                # Sample offspring
                offspring = []
                for k in range(N):
                    z = np.random.standard_normal(dim)
                    y = C_sqrt @ z
                    offspring.append(x0 + sigma * y)

                # Evaluate offspring (respect budget/time)
                off_f = []
                for v in offspring:
                    if calls_used >= budget or time.time() > deadline:
                        break
                    off_f.append(counted_func(v))

                if len(off_f) < N:
                    break

                # Sort by fitness
                order = np.argsort(off_f)
                sorted_off = [offspring[i] for i in order]
                sorted_f = [off_f[i] for i in order]

                # Update mean (weighted recombination)
                y_w = np.zeros(dim)
                for k in range(mu):
                    y_w += w[k] * (sorted_off[k] - x0) / sigma
                x0 = x0 + sigma * y_w

                # Update step-size (sigma)
                y_sigma = C_inv_sqrt @ y_w
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chiN - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)

                # Update evolution paths and covariance
                y_c = y_w
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * y_c
                H = np.outer(p_c, p_c)
                H_h = (1 - c_1) * C
                H_h += c_1 * (H - (1 - c_1) * C)
                H_h += c_mu / c_1 * sum(w[k] * np.outer(sorted_off[k] - x0, sorted_off[k] - x0) for k in range(mu))
                C = (1 + c_1 * (mueff - 1 - c_1)) * C + H_h
                # Periodically recompute Cholesky and inverse sqrt
                recompute_chol()

        return best_x, best_f
# EVOLVE-BLOCK-END
