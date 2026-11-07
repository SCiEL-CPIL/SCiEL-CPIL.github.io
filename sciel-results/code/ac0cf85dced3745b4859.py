"""Optimizer for BBOB f4 (Büche-Rastrigin) using hybrid strategy."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer combining:
    1. CMA-ES for global exploration (first ~60% of budget)
    2. (1+1)-ES with adaptive step size for local exploitation
    3. Nelder-Mead for final polishing
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin before 300s limit

        # Phase 1: CMA-ES for global exploration
        # Phase 2: (1+1)-ES for local refinement
        # Phase 3: Nelder-Mead for final polish

        # --- CMA-ES ---
        # Initialize
        rng = np.random.default_rng(42)
        x = rng.uniform(-5, 5, dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(dim)))  # population size
        sigma = 1.0
        c_sigma = (lambda_ + 2) / (dim + lambda_ + 5)
        c_c = (4 + lambda_ / dim) / (dim + 4 + 2 * lambda_ / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + lambda_)
        c_mu = min(1 - c_1, 2 * ((lambda_ / 2 + 1) / (dim + 2) - 1) / ((dim + 2) ** 2 + lambda_))
        c_mu = max(c_mu, 0.1)

        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # CMA-ES loop for first 60% of budget
        cma_budget = int(0.6 * budget)
        for _ in range(cma_budget - 1):
            if time.time() > deadline:
                break
            if calls >= budget:
                break

            # Generate offspring
            W = 1.0 / dim  # simplified weights
            mu = lambda_ // 2
            mu_eff = 1.0 / (W * mu)

            # Update covariance adaptation
            C_sqrt_inv = np.linalg.inv(np.linalg.cholesky(C + 1e-12 * np.eye(dim)))

            # Sample
            ys = []
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                y = x + sigma * C_sqrt_inv.T @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                ys.append((fy, y))

            # Sort
            ys.sort(key=lambda t: t[0])

            # Update mean
            w = np.zeros(lambda_)
            for i in range(mu):
                w[i] = 1.0 / mu
            w_sum = w.sum()

            x_new = sum(w[i] * ys[i][1] for i in range(mu))

            # Update step-size control
            d_sigma = 1 + 2 * max(0, (mu_eff / dim - 1) / (dim + 2))
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv.T @ (x_new - x) / sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = min(max(sigma, 1e-10), 10.0)

            # Update covariance matrix
            diff = x_new - x
            pc = (1 - c_c) * pc + c_c * diff / sigma
            C = (1 + c_1 * (2 - c_1) + c_mu * (2 - c_mu) - c_1 - c_mu) * C
            C += c_1 * (pc @ pc.T - C)
            C += c_mu / w_sum * sum(w[i] * (np.outer(ys[i][1] - x, ys[i][1] - x)) for i in range(mu))

            # Ensure positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            x = x_new
            fx = min(fy for fy, _ in ys)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # --- Phase 2: (1+1)-ES for local refinement ---
        # Use remaining budget
        es_budget = budget - calls
        if es_budget > 10 and time.time() < deadline:
            x_es = best_x.copy()
            fx_es = best_fx
            sigma_es = 0.5
            for _ in range(es_budget):
                if time.time() > deadline:
                    break
                y = np.clip(x_es + sigma_es * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                calls += 1
                if fy < fx_es:
                    x_es, fx_es = y, fy
                    sigma_es *= 1.2
                    if fx_es < best_fx:
                        best_fx = fx_es
                        best_x = x_es.copy()
                else:
                    sigma_es *= 0.85

        # --- Phase 3: Nelder-Mead for final polish ---
        nm_budget = min(50, max(10, budget - calls))
        if nm_budget > 5 and time.time() < deadline:
            try:
                from scipy.optimize import minimize
                res = minimize(func, best_x, method='Nelder-Mead',
                               options={'maxiter': nm_budget, 'xatol': 1e-6, 'fatol': 1e-8})
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
