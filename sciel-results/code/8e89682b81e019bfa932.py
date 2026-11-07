"""Optimizer for BBOB f17 (Schaffers F7) using CMA-ES + local polish."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES with local refinement (Nelder-Mead + L-BFGS-B) for 5D multimodal optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 320  # 10s safety margin before 330s kill

        def time_ok():
            return time.time() < deadline

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = 20  # population size
        mu = lambda_ // 2  # recombination weight count
        # Weights
        weights = np.log(np.arange(mu, 0, -1)) - np.log(np.arange(mu, 0, -1)).sum()
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        # CMA-ES step size adaptation
        c_sigma = (mueff + 2) / ((dim + 1) ** 2 + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial point: center of box
        x = np.zeros(dim)
        sigma = 2.0  # initial step size
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_inv_sqrt = np.eye(dim)

        best_x = x.copy()
        best_f = func(best_x)
        calls = 1

        def clip(x):
            return np.clip(x, -5, 5)

        # Phase 1: CMA-ES global search
        cma_budget = int(budget * 0.75)  # 75% for CMA-ES
        polish_budget = budget - cma_budget  # 25% for local polish

        for gen in range(cma_budget // lambda_ + 1):
            if not time_ok() or calls >= cma_budget:
                break

            # Generate population
            offspring = []
            for i in range(lambda_):
                if not time_ok() or calls >= cma_budget:
                    break
                z = np.random.standard_normal(dim)
                y = clip(x + sigma * C_sqrt @ z)
                fy = func(y)
                calls += 1
                offspring.append((y, fy))

            if len(offspring) < mu:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[1])
            selected = offspring[:mu]

            # Update best
            if selected[0][1] < best_f:
                best_f = selected[0][1]
                best_x = selected[0][0].copy()

            # Weighted mean
            x_new = sum(w * y for w, (y, _) in zip(weights, selected))

            # Update covariance matrix
            y_vec = (x_new - x) / sigma
            C_old = C
            C = (1 + c_1 * (c_1 - c_mu)) * C + c_1 * np.outer(y_vec, y_vec)
            for w, (y, _) in zip(weights, selected):
                v = (y - x) / sigma
                C += c_mu * w * np.outer(v, v)

            # Update step size
            C_sqrt = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
            C_inv = np.linalg.inv(C)
            C_inv_sqrt = np.linalg.cholesky(C_inv + 1e-12 * np.eye(dim))

            norm_y = np.linalg.norm(C_inv_sqrt @ y_vec)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_vec
            sigma *= np.exp(c_sigma / d_sigma * (norm_y / chi_n - 1))
            sigma = max(sigma, 1e-12)
            sigma = min(sigma, 10.0)

            # Update evolution path
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * C_inv_sqrt @ y_vec
            x = clip(x_new)

            # Ensure C is positive definite
            try:
                C_sqrt = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
            except np.linalg.LinAlgError:
                C += 1e-10 * np.eye(dim)
                C_sqrt = np.linalg.cholesky(C)

        # Phase 2: Local refinement
        # Nelder-Mead polish
        if calls < budget and time_ok():
            res = minimize(
                func, best_x, method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-14, 'adaptive': True}
            )
            calls += res.nfev
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()

        # L-BFGS-B polish
        if calls < budget and time_ok():
            def grad(x):
                # Numerical gradient
                g = np.zeros(dim)
                eps = 1e-7
                for i in range(dim):
                    xp = x.copy()
                    xm = x.copy()
                    xp[i] += eps
                    xm[i] -= eps
                    g[i] = (func(np.clip(xp, -5, 5)) - func(np.clip(xm, -5, 5))) / (2 * eps)
                return g

            # Use a subset of remaining budget for L-BFGS
            remaining = budget - calls
            maxiter = min(100, remaining // (dim + 1))
            if maxiter > 5:
                res2 = minimize(
                    func, best_x, jac=grad, method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': maxiter, 'gtol': 1e-12, 'ftol': 1e-14}
                )
                calls += res2.nfev
                if res2.fun < best_f:
                    best_f = res2.fun
                    best_x = res2.x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
