"""Optimized CMA-ES + (1+1)-ES for BBOB f11 (Discus)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with (1+1)-ES polish, deadline-guarded."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit

        rng = np.random.default_rng(42)
        n = self.dim

        # Initial point
        x0 = np.zeros(n)
        best_x = x0.copy()
        best_f = func(best_x)
        calls = 1

        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * log(n))
        lam = 4 + int(3 * np.log(n)) if n > 1 else 4
        # Initial sigma: 2.0 (box is [-5,5]^n, minimizer in [-4,4]^n)
        sigma = 2.0

        # CMA-ES state
        mean = x0.copy()
        C = np.eye(n)
        C_inv = np.eye(n)
        C_sqrt = np.eye(n)
        C_inv_sqrt = np.eye(n)
        # For n=5, we can maintain Cholesky-like updates, but for simplicity
        # with small n, we'll use a simplified approach

        # Evolution paths
        ps = np.zeros(n)
        pc = np.zeros(n)
        # Step sizes
        cs = (1.5 + n / 2.0) / (n + 1.5 + 3.0)
        cmu = 2 / ((n + 1.3)**2 + cs)
        c1 = (2 + 2 / (n + 1.3)) / ((n + 2)**2 + 2)
        cmu = min(2 * (1 - c1) / ((n + 2)**2 + 2), 0.5 * c1)
        cc = (4 + n / cs) / (n + 4 + 2 / cs)
        ds = 1 + 2 * max(0, np.sqrt((n - 1) / (n + 1)) - 1) + cs
        # Chi-lambda: expected ||N(0,I)|| for standard normal
        chi_lam = np.sqrt(lam) * (1 - 1 / (4 * lam) + 1 / (21 * lam**2))

        def cholesky_inv_sqrt(A):
            """Compute inverse square root of positive definite matrix A."""
            # For small n, use eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(A)
            eigvals = np.clip(eigvals, 1e-10, None)
            return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

        def mat_sqrt(A):
            """Compute square root of positive definite matrix A."""
            eigvals, eigvecs = np.linalg.eigh(A)
            eigvals = np.clip(eigvals, 1e-10, None)
            return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

        C_sqrt = mat_sqrt(C)
        C_inv_sqrt = cholesky_inv_sqrt(C)

        # Generate initial population
        def generate_population():
            nonlocal C_sqrt, C_inv_sqrt, mean, sigma
            ys = rng.standard_normal((lam, n))
            # Transform to current distribution
            xs = np.array([mean + sigma * (C_sqrt @ y.T).T for y in ys])
            return xs, ys

        # CMA-ES main loop
        while calls < self.budget:
            if time.time() > deadline:
                break

            # Generate and evaluate population
            xs, ys = generate_population()
            fs = np.array([func(xi) for xi in xs])
            calls += lam

            if time.time() > deadline:
                break

            # Sort by fitness
            order = np.argsort(fs)
            xs = xs[order]
            ys = ys[order]
            fs = fs[order]

            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = xs[0].copy()

            # Update mean
            w = np.log(np.arange(1, lam // 2 + 1))
            w = w / w.sum()
            mu_eff = 1 / np.sum(w**2)
            mean_old = mean.copy()
            mean = np.sum(w[:, None] * xs[:lam // 2], axis=0)

            # Update evolution path ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (C_inv_sqrt @ (mean - mean_old).T / sigma).T

            # Update sigma
            sigma *= np.exp((1 / ds) * (np.linalg.norm(ps) / chi_lam - 1))
            sigma = np.clip(sigma, 1e-10, 1e10)

            # Update evolution path pc
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (calls - lam)))) < (1.4 + 2 / (n + 1)) * chi_lam
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) * (mean - mean_old).T / sigma

            # Update covariance matrix C
            delta = hsig * (1 - cc) * cc
            rank_one = np.outer(pc, pc)
            rank_mu = np.sum(w[:, None, None] * np.outer(ys[:lam // 2], ys[:lam // 2]), axis=0)
            C = (1 + c1 * delta - c1 - cmu) * C + c1 * (rank_one + np.eye(n)) + cmu * rank_mu

            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            C_sqrt = mat_sqrt(C)
            C_inv_sqrt = cholesky_inv_sqrt(C)

        # (1+1)-ES polish with remaining budget
        if calls < self.budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma_es = 0.1
            for _ in range(self.budget - calls):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma_es * rng.standard_normal(n), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma_es *= 1.2
                else:
                    sigma_es *= 0.85
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
