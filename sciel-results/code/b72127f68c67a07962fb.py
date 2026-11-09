"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with proper implementation and (1+1)-ES polish for
    Griewank-Rosenbrock hybrid in 5D."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = 325.0  # 5s safety margin before 330s kill
        dim = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf

        rng = np.random.default_rng(42)

        # Initial exploration: 10 random points + center
        init_budget = 10
        candidates = [np.zeros(dim)]
        for _ in range(init_budget - 1):
            candidates.append(rng.uniform(-4, 4, dim))

        cand_vals = [func(np.clip(c, -5, 5)) for c in candidates]
        best_idx = int(np.argmin(cand_vals))
        x = np.clip(candidates[best_idx], -5, 5).copy()
        fx = cand_vals[best_idx]
        used = init_budget

        if fx < best_f:
            best_f = fx
            best_x = x.copy()

        # CMA-ES parameters (lambda=12, mu=6)
        lam = 12
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        sigma = 1.0
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Reserve 50 evals for polish
        polish_budget = 50
        cma_budget = max(0, total_budget - used - polish_budget)
        max_iters = cma_budget // lam

        for it in range(max_iters):
            if time.time() > deadline or used >= total_budget - polish_budget:
                break

            # Sample lambda points from N(0, C) using eigendecomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            sqrt_eigvals = np.sqrt(eigvals)

            Y = np.zeros((lam, dim))
            for i in range(lam):
                z = rng.standard_normal(dim)
                Y[i] = eigvecs @ (sqrt_eigvals * z)

            # Evaluate
            X = np.clip(x[None, :] + sigma * Y, -5, 5)
            F = np.array([func(xi) for xi in X])
            used += lam

            # Update best
            for i in range(lam):
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = X[i].copy()

            # Sort by fitness
            order = np.argsort(F)
            Y_sorted = Y[order]
            X_sorted = X[order]
            F_sorted = F[order]

            # Mean update
            x = x + sigma * np.sum(w[:, None] * Y_sorted[:mu], axis=0)
            x = np.clip(x, -5, 5)

            # Covariance matrix adaptation (standard CMA-ES)
            ys = Y_sorted[:mu]
            H = np.zeros((dim, dim))
            for i in range(mu):
                H += w[i] * np.outer(ys[i], ys[i])

            # Update path variables
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * np.sum(w[:mu] * ys, axis=0)

            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * np.sum(w[:mu] * ys, axis=0)

            # Covariance update
            factor = (1 - c_c - c_sigma * (1 - c_sigma) + c_sigma ** 2 * (1 - 1.0 / mu_eff))
            C_new = factor * C + c_sigma * (1 - 1.0 / mu_eff) * H
            C_new = (C_new + C_new.T) / 2

            # Ensure PSD
            eigvals_C, eigvecs_C = np.linalg.eigh(C_new)
            if np.min(eigvals_C) < 1e-12:
                C_new = eigvecs_C @ np.diag(np.clip(eigvals_C, 1e-10, None)) @ eigvecs_C.T

            C = C_new

            # Step-size adaptation
            norm_p = np.linalg.norm(p_sigma)
            sigma *= np.exp((1 / (5 * dim + 5)) * (norm_p / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 20.0)

        # (1+1)-ES polish with remaining budget
        if time.time() < deadline and used < total_budget:
            x = best_x.copy()
            fx = best_f
            sigma = max(0.02, sigma * 0.1)
            polish_left = total_budget - used
            for _ in range(polish_left):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                used += 1
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.8
        return best_x, best_f
# EVOLVE-BLOCK-END
