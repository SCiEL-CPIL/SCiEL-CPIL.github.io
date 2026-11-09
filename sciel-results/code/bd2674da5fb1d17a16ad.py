"""Rewritten optimizer for BBOB f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import time
import math
import numpy as np


class Optimizer:
    """CMA-ES style adaptive-covariance optimizer with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin under 300 s

        # Population size (CMA-ES default)
        lambda_ = max(4, int(4 + 3 * math.log(d)))
        # Strategy parameters
        cs = (lambda_ + 2) / (d + lambda_ + 5)
        ds = 1 + 2 * max(0, math.sqrt((lambda_ - 1) / (d + 1)) - 1) + cs
        c1 = (2 + 2.0 / ((d + 1.3) ** 2)) / ((d + 4) ** 1.5)
        cmu = min(1 - c1,
                  2 * (lambda_ * 0.5 - 1 + 1.0 / lambda_) /
                  ((d + 2) ** 2 + 2))
        chi_d = math.sqrt(d) * (1 - 1.0 / (4 * d) + 1.0 / (21 * d * d))

        # Initial state
        x = np.zeros(d)
        sigma = 1.5
        B = np.eye(d)
        D = np.ones(d)
        ps = np.zeros(d)
        hsig = (1 + 2.0 / (d + 1)) * chi_d
        c_sigma = max(1.0 / ds, 0.001)
        c_sigma = min(c_sigma, 0.5)

        # Initial sampling: uniform over the box, evaluate, keep best
        rng = np.random.default_rng(0)
        X0 = rng.uniform(-5, 5, (lambda_, d))
        F0 = np.array([func(xi) for xi in X0])
        order0 = np.argsort(F0)
        x = X0[order0[0]].copy()
        fx_best = F0[order0[0]]
        calls = lambda_

        # Main loop
        while calls < budget and time.time() < deadline:
            # Sample
            Z = rng.standard_normal((lambda_, d))
            X = np.clip(x[None, :] + sigma * (Z @ (B * D)), -5, 5)
            F = np.array([func(xi) for xi in X])
            calls += lambda_

            order = np.argsort(F)
            mu = lambda_ // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights = weights / weights.sum()
            mueff = 1.0 / weights.sum()

            # Update best-so-far
            if F[order[0]] < fx_best:
                fx_best = F[order[0]]
                x = X[order[0]].copy()

            # Mean update
            y_w = np.zeros(d)
            for i in range(mu):
                y_w += weights[i] * (X[order[i]] - x)
            x = x + y_w

            # Sigma update (path adaptation)
            ps = (1 - cs) * ps + math.sqrt(cs * (2 - cs) * mueff) * (B.T @ (y_w / sigma))
            sigma *= math.exp(c_sigma / ds * (np.linalg.norm(ps) / chi_d - 1))
            sigma = min(sigma, 1000.0)
            sigma = max(sigma, 1e-12)

            # Covariance update
            if np.linalg.norm(ps) < (1.4 + 1.0 / ds) * hsig:
                hsig = 1.0
            else:
                hsig = 0.995

            C = (B * D) @ B.T
            C = (1 - c1 - cmu) * C + c1 * (np.outer(ps, ps) + (hsig - 1) * c_sigma * (2 - c_sigma) * C)
            for i in range(mu):
                zi = (X[order[i]] - x) / sigma
                C += cmu * weights[i] * np.outer(zi, zi)

            # Re-orthonormalize C (eigh)
            try:
                evals, evecs = np.linalg.eigh(C)
                evals = np.clip(evals, 1e-12, None)
                D = np.sqrt(evals)
                B = evecs
            except np.linalg.LinAlgError:
                B = np.eye(d)
                D = np.ones(d) * max(sigma, 1e-6)

        return x, fx_best
# EVOLVE-BLOCK-END
