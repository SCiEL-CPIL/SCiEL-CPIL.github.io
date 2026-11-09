import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = 310.0
        dim = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = 0

        rng = np.random.default_rng(12345)

        def evaluate(x):
            nonlocal best_f, best_x, call_count
            if call_count >= total_budget or time.time() > deadline:
                return best_f + 1e6
            x = np.clip(np.asarray(x, dtype=float), -5, 5)
            val = func(x)
            call_count += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Stratified initialization
        n_init = min(20, total_budget)
        init_points = []
        # Center
        init_points.append(np.zeros(dim))
        # Axis-aligned points
        for i in range(dim):
            for s in [-3, 3]:
                p = np.zeros(dim)
                p[i] = s
                init_points.append(p)
        # Random points
        while len(init_points) < n_init:
            init_points.append(rng.uniform(-4, 4, dim))
        
        init_points = init_points[:n_init]
        for p in init_points:
            evaluate(p)

        # Phase 2: CMA-ES with adaptive restarts
        # Multiple restarts to handle multimodality
        restarts = 3
        budget_per_restart = (total_budget - call_count) // restarts if call_count < total_budget else 0
        polish_budget = 60
        cma_budget = max(0, total_budget - call_count - polish_budget)
        budget_per_restart = cma_budget // restarts if cma_budget > 0 else 0

        for restart in range(restarts):
            if time.time() > deadline or call_count >= total_budget - polish_budget:
                break
            if budget_per_restart <= 0:
                break

            # Choose starting point: best from previous, or random
            if restart == 0:
                x = best_x.copy() if best_x is not None else np.zeros(dim)
                sigma = 1.5
            else:
                # Perturb from best or random
                if rng.random() < 0.5 and best_x is not None:
                    x = best_x + rng.normal(0, 1.0, dim)
                    x = np.clip(x, -5, 5)
                    sigma = 0.8
                else:
                    x = rng.uniform(-4, 4, dim)
                    sigma = 1.5

            # CMA-ES parameters
            lam = 15  # Population size
            mu = lam // 2
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)

            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            L = np.linalg.cholesky(C)

            max_iters = budget_per_restart // lam
            if max_iters < 1:
                max_iters = 1

            for it in range(max_iters):
                if time.time() > deadline or call_count >= total_budget - polish_budget:
                    break

                # Generate offspring
                Y = np.zeros((lam, dim))
                for i in range(lam):
                    z = rng.standard_normal(dim)
                    Y[i] = L @ z

                # Evaluate
                X = np.clip(x[None, :] + sigma * Y, -5, 5)
                F = np.array([evaluate(xi) for xi in X])

                # Check if we've run out of budget
                if call_count >= total_budget - polish_budget:
                    break

                # Sort
                order = np.argsort(F)
                F_sorted = F[order]
                Y_sorted = Y[order]

                # Update mean
                x = x + sigma * np.sum(w[:, None] * Y_sorted[:mu], axis=0)
                x = np.clip(x, -5, 5)

                # Update covariance
                ys = Y_sorted[:mu]
                mean_y = np.sum(w[:, None] * ys, axis=0)

                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * mean_y

                H = np.zeros((dim, dim))
                for i in range(mu):
                    H += w[i] * np.outer(ys[i], ys[i])

                coeff_c = (1 - c_c) * (1 - c_sigma)
                coeff_pc = c_c * (1 - c_c) * d_sigma
                coeff_h = c_sigma * (1 - 1.0 / mu_eff)

                C_new = coeff_c * C + coeff_pc * np.outer(p_c, p_c) + coeff_h * H
                C_new = (C_new + C_new.T) / 2

                # Ensure positive definite
                eigvals, eigvecs = np.linalg.eigh(C_new)
                min_eig = np.min(eigvals)
                if min_eig < 1e-12:
                    eigvals = np.clip(eigvals, 1e-12, None)
                    C_new = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C = C_new

                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(dim)
                    L = np.eye(dim)

                # Step-size control
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * mean_y
                sigma *= np.exp((1.0 / (5 * dim + 5)) * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)

        # Phase 3: Local polish with Nelder-Mead
        if time.time() < deadline and call_count < total_budget:
            x_start = best_x.copy() if best_x is not None else np.zeros(dim)
            max_calls = total_budget - call_count

            def nm_func(x):
                nonlocal best_f, best_x
                x = np.clip(np.asarray(x, dtype=float), -5, 5)
                val = func(x)
                call_count += 1
                if val < best_f:
                    best_f = val
                    best_x = x.copy()
                return val

            try:
                res = minimize(nm_func, x_start, method='Nelder-Mead',
                               options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12,
                                        'adaptive': True})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

            # Second polish from a slightly perturbed point
            if time.time() < deadline and call_count < total_budget:
                x_start2 = best_x + rng.normal(0, 0.1, dim)
                x_start2 = np.clip(x_start2, -5, 5)

                def nm_func2(x):
                    nonlocal best_f, best_x
                    x = np.clip(np.asarray(x, dtype=float), -5, 5)
                    val = func(x)
                    call_count += 1
                    if val < best_f:
                        best_f = val
                        best_x = x.copy()
                    return val

                try:
                    res2 = minimize(nm_func2, x_start2, method='Nelder-Mead',
                                    options={'maxiter': 200, 'xatol': 1e-10, 'fatol': 1e-12,
                                             'adaptive': True})
                    if res2.fun < best_f:
                        best_f = res2.fun
                        best_x = np.clip(res2.x, -5, 5)
                except Exception:
                    pass

        return best_x, best_f
