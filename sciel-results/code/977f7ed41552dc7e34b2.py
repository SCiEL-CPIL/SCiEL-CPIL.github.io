"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for BBOB-style Rosenbrock in 5D."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270  # 30s safety margin before 300s limit
        n = self.dim
        budget = self.budget

        # Population size for CMA-ES
        lam = 4 + int(3 * np.log(n))  # 4 + 3*log(5) ≈ 9
        mu = lam // 2

        # Initialize
        rng = np.random.default_rng(42)
        x = rng.uniform(-4, 4, n)  # Start in interior
        fx = func(x)
        best_x, best_fx = x.copy(), fx

        # CMA-ES parameters
        sigma = 2.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        cc = (4 + mu / n) / ((n + 4) + 2 * mu / n)
        cs = (mu + 2) / (n + mu + 5)
        cmu = min(1 - 2 * cc, (mu - 2 + 1 / mu) / (n + 2) - 2 * cc)
        damps = (1 + 2 * max(0, np.sqrt((mu - 1) / (n + 1)) - 1)) * cs
        damps = min(damps, 2 - cc)

        calls = 1

        while calls < budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            w = np.log(lam + 0.5) - np.log(np.arange(1, lam + 1))
            w /= w.sum()
            mueff = 1 / np.sum(w[:mu] ** 2)

            # Sample
            Y = np.zeros((lam, n))
            F = np.zeros(lam)
            for i in range(lam):
                z = rng.standard_normal(n)
                Y[i] = x + sigma * (C @ z)
                F[i] = func(Y[i])
                calls += 1
                if calls >= budget:
                    break
                if time.time() > deadline:
                    break

            # Update best
            for i in range(lam):
                if F[i] < best_fx:
                    best_fx = F[i]
                    best_x = Y[i].copy()

            if calls >= budget:
                break

            # Sort by fitness
            idx = np.argsort(F)
            Y = Y[idx]
            F = F[idx]

            # Weighted mean
            m = np.sum(w[:mu, None] * Y[:mu], axis=0)

            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C @ ((m - x) / sigma))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls + 1)))) / \
                   np.sqrt(1 - (1 - cs) ** (2 * (calls + 1)))
            hsig = min(hsig, 1.4 + 2 / (n + 1))

            pc = (1 - cc) * pc + hsig * cc * (m - x) / sigma

            # Update covariance matrix
            delta_h = (1 - hsig) * cc * (2 - cc)
            C = (1 + cmu * delta_h - cmu) * C
            for i in range(mu):
                yi = (Y[i] - x) / sigma
                C += cmu * w[i] * np.outer(yi, yi)
                C -= cmu * delta_h * w[i] * np.outer(pc, pc)

            # Update position
            x = m

            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(2 * n) - 1))

            # Clip x to bounds
            x = np.clip(x, -5, 5)

            # If x is at boundary, nudge inward slightly
            if np.any(x <= -4.99) or np.any(x >= 4.99):
                x = np.clip(x, -4.5, 4.5)

            # Update current best if improved
            if F[0] < best_fx:
                best_fx = F[0]
                best_x = x.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
