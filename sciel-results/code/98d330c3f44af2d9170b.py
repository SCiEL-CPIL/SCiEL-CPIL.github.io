"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time budget guard for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s limit
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(n))  # for n=5, lam=10
        # Initial step size
        sigma = 2.0
        # Initial mean - start near center but slightly random
        rng = np.random.default_rng(42)
        m = np.zeros(n)

        # Covariance matrix
        C = np.eye(n)
        # Step sizes for C update
        cs = (n + 2) / (n + 8)
        ds = 1 + 2 * max(0, np.sqrt((n - 1) / (n + 3)) - 1) + cs
        c1 = (2 / ((n + 1.3) ** 2 + lam)) * (1 - cs)
        cmu = min(1 - c1, 2 * ((n - 2 + 1 / n) / ((n + 2) ** 2 + lam)))
        # Evolution path for covariance
        pc = np.zeros(n)
        # Evolution path for step size
        ps = np.zeros(n)
        # Inverse square root of C
        C_eigvals, C_eigvecs = np.linalg.eigh(C)
        C_eigvals = np.maximum(C_eigvals, 1e-20)
        C_sqrt_inv = C_eigvecs @ np.diag(1.0 / np.sqrt(C_eigvals)) @ C_eigvecs.T
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # Evaluate initial point
        x0 = np.clip(m, -5, 5)
        best_x = x0.copy()
        best_f = func(x0)
        calls = 1

        # Precompute weights for weighted mean
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights ** 2)

        while calls < budget:
            # Check time
            if time.time() > deadline:
                break

            # Sample population
            pops = np.empty((lam, n))
            for i in range(lam):
                z = rng.standard_normal(n)
                pops[i] = m + sigma * (C_sqrt_inv @ z)
                # Clip to box
                pops[i] = np.clip(pops[i], -5, 5)

            # Evaluate
            fs = np.empty(lam)
            for i in range(lam):
                if time.time() > deadline:
                    # Fill remaining with best known
                    fs[i:] = best_f
                    break
                fs[i] = func(pops[i])
                calls += 1
                if fs[i] < best_f:
                    best_f = fs[i]
                    best_x = pops[i].copy()

            # Sort by fitness
            order = np.argsort(fs)
            pops_sorted = pops[order]
            fs_sorted = fs[order]

            # Compute weighted mean
            m_new = np.sum(weights[:, None] * pops_sorted[:mu], axis=0)

            # Update evolution path for covariance
            pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mueff) * C_sqrt_inv @ (m_new - m) / sigma

            # Update evolution path for step size
            ps = (1 - ds) * ps + np.sqrt(ds * (2 - ds) * mueff) * (m_new - m) / sigma

            # Update sigma
            hsig = min(1.0, np.linalg.norm(ps) / (1 - 0.5 ** (calls / (lam * ds))) / chi_n)
            sigma *= np.exp(cs / lam * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100.0)

            # Update covariance matrix
            if hsig < 1.4:
                delta_hsig = (1 - hsig) * cs * (2 - cs)
                C = (1 - c1 - cmu * mueff / 2 + delta_hsig) * C
                C += c1 * (np.outer(pc, pc) - (1 - delta_hsig) * C)
                # Rank-mu update
                y = (m_new - m) / sigma
                for i in range(mu):
                    C += cmu * weights[i] * (pops_sorted[i] - m) @ (pops_sorted[i] - m).T / sigma ** 2

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            C_eigvals, C_eigvecs = np.linalg.eigh(C)
            C_eigvals = np.maximum(C_eigvals, 1e-20)
            C = C_eigvecs @ np.diag(C_eigvals) @ C_eigvecs.T
            C_sqrt_inv = C_eigvecs @ np.diag(1.0 / np.sqrt(C_eigvals)) @ C_eigvecs.T

            # Update mean
            m = m_new
            # Keep m within bounds
            m = np.clip(m, -5, 5)

            # If we've made good progress, tighten around best
            if calls % 50 == 0 and best_f < 10:
                # Blend mean toward best
                m = 0.5 * m + 0.5 * best_x

        return best_x, best_f
# EVOLVE-BLOCK-END
