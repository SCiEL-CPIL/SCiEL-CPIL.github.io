"""Black-box optimizer for BBOB f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np


class Optimizer:
    """Multi-restart CMA-ES with an early random-sampling phase."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before the 330s kill
        dim = self.dim
        budget = self.budget

        # Deterministic RNG for reproducibility
        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf
        calls = 0

        def time_ok():
            return (time.time() - t0) < deadline and calls < budget

        # ---------- Phase 1: cheap random sampling for fast early progress ----------
        # Random sampling gives a big initial drop in the log-error curve,
        # which is where AOCC is most sensitive.
        n_random = min(30, budget)
        for _ in range(n_random):
            if not time_ok():
                break
            x = rng.uniform(-5.0, 5.0, dim)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            # Fallback: at least one call
            x = np.zeros(dim)
            best_f = func(x)
            best_x = x.copy()
            calls += 1
            return best_x, best_f

        # ---------- Phase 2: Multi-restart CMA-ES ----------
        # CMA-ES parameters
        lam = 4 + 3 * int(math.ceil(3.0 * math.log(dim)))  # population size
        mu = lam // 2  # number of parents
        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        # Learning rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        # Initial sigma
        sigma0 = 1.0

        # Track progress for restart decisions
        no_improve = 0
        restart_threshold = int(1.5 * dim * (1 + math.log(dim)))

        while time_ok():
            # Initialize CMA-ES state from best_x
            m = best_x.copy()
            sigma = sigma0
            B = np.eye(dim)
            D = np.ones(dim)
            ps = np.zeros(dim)
            hs = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # Stagnation counter for this restart
            local_no_improve = 0

            while time_ok():
                # Generate population
                if lam > budget - calls:
                    break
                # Sample
                zs = rng.standard_normal((lam, dim))
                # Transform to x
                X = np.empty((lam, dim))
                for i in range(lam):
                    X[i] = m + sigma * (zs[i] @ (B * D))
                # Clip to box
                X = np.clip(X, -5.0, 5.0)

                # Evaluate
                F = np.empty(lam)
                for i in range(lam):
                    if not time_ok():
                        break
                    F[i] = func(X[i])
                    calls += 1

                # Sort by fitness
                order = np.argsort(F)
                X_sorted = X[order]
                F_sorted = F[order]

                # Update best
                if F_sorted[0] < best_f:
                    best_f = F_sorted[0]
                    best_x = X_sorted[0].copy()
                    local_no_improve = 0
                    no_improve = 0
                else:
                    local_no_improve += 1
                    no_improve += 1

                # Update mean
                m_old = m.copy()
                m = np.sum(weights[:, None] * X_sorted[:mu], axis=0)

                # Update evolution paths
                inv_sqrtD = 1.0 / np.sqrt(D)
                dmean = (m - m_old) / sigma
                ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * (B.T @ (inv_sqrtD * dmean))
                hsig = (np.linalg.norm(ps) / math.sqrt(1 - (1 - cs) ** (calls + 1))) <= \
                       (1.4 + 2 / (dim + 1)) * chi_n
                hs = (1 - cs) * hs + hsig * math.sqrt(cs * (2 - cs) * mueff) * (B @ dmean)

                # Update covariance matrix
                delta_hs = (1.0 - hsig) * cs * (2 - cs)
                C = B @ (np.outer(B.T, B.T) * (D ** 2))  # This is O(dim^3), fine for dim=5
                # Simpler: update C directly
                C = B @ np.diag(D ** 2) @ B.T
                C = (1 + c1 * delta_hs - c1 - cmu) * C + \
                    c1 * (np.outer(ps, ps) + (delta_hs - 1 + cs ** 2) * np.eye(dim)) + \
                    cmu * np.sum(weights[:, None, None] * np.outer(X_sorted[:mu] - m_old, X_sorted[:mu] - m_old), axis=0)

                # Eigendecomposition for B and D
                try:
                    D2, B = np.linalg.eigh(C)
                    D2 = np.sqrt(np.maximum(D2, 1e-20))
                    D = D2
                except np.linalg.LinAlgError:
                    break

                # Update sigma (CSA)
                sigma *= math.exp(cs / (dim + 2) * (np.linalg.norm(ps) / chi_n - 1))

                # Restart conditions
                if sigma < 1e-8 or sigma > 1e6:
                    break
                if local_no_improve > restart_threshold:
                    break
                if np.linalg.norm(m - m_old) / (sigma + 1e-30) < 1e-12:
                    break

            # After a restart, if we've made no progress overall for a while, break
            if no_improve > 3 * restart_threshold:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
