"""Hybrid CMA-ES optimizer for Rastrigin f3 on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: (1+1)-ES exploration + CMA-ES + Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 310.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)

        # Phase 1: (1+1)-ES for fast initial exploration
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1
        sigma = 2.0

        # (1+1)-ES with 1/5 success rule
        es_budget = max(50, budget // 5)
        while calls < es_budget:
            if time.time() > deadline - 5:
                break
            y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            calls += 1
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.15
            else:
                sigma *= 0.88
            if fy < best_fx:
                best_fx = fy
                best_x = y.copy()

        # Phase 2: CMA-ES for exploitation
        lam = max(4, int(4 + 3 * np.log(dim)))  # population size
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()

        x_cma = best_x.copy()
        sigma_cma = max(0.5, sigma * 0.5)
        C = np.eye(dim)
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.clip(eigvals, 1e-12, None)
        C_sqrt = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
        C_inv_sqrt = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        cma_calls = 0
        reserve = 30  # reserve for polish

        while calls < budget - reserve:
            if time.time() > deadline - 5:
                break

            # Generate population
            X = np.zeros((lam, dim))
            F = np.zeros(lam)
            for i in range(lam):
                if calls >= budget - reserve:
                    break
                step = sigma_cma * (C_sqrt @ rng.standard_normal(dim))
                X[i] = np.clip(x_cma + step, -5, 5)
                F[i] = func(X[i])
                calls += 1
                if F[i] < best_fx:
                    best_fx = F[i]
                    best_x = X[i].copy()

            if calls >= budget - reserve:
                break

            # Sort by fitness
            idx = np.argsort(F)
            X_sorted = X[idx]
            F_sorted = F[idx]

            # Weighted recombination
            x_old = x_cma.copy()
            x_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
            x_new = np.clip(x_new, -5, 5)

            # Update sigma (CSA)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma_cma * (C_inv_sqrt @ (x_new - x_old))
            norm_ps = np.linalg.norm(p_sigma)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            sigma_cma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
            sigma_cma = np.clip(sigma_cma, 1e-12, 1e6)

            # Update C (rank-one and rank-mu updates)
            remaining = max(1, (budget - reserve) - calls)
            h_sigma = norm_ps / np.sqrt(1 - (1 - c_sigma) ** remaining) < (1 + 2 / (dim + 1)) * chi_n
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x_old) / sigma_cma

            # Rank-one update
            C += c_1 * (np.outer(p_c, p_c) / sigma_cma ** 2 - 1) * C
            # Rank-mu update
            y = (X_sorted[:mu] - x_old) / sigma_cma
            C += c_mu * np.sum(weights[:, None, None] * np.outer(y[:, :, None], y[:, None, :]), axis=0)

            # Update C_sqrt via eigendecomposition every 5 iterations
            if cma_calls % 5 == 0:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C_sqrt = (eigvecs * np.sqrt(eigvals)) @ eigvecs.T
                C_inv_sqrt = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T

            x_cma = x_new
            cma_calls += 1

        # Phase 3: Nelder-Mead local polish
        if calls < budget - 5 and time.time() < deadline - 5:
            x_nm = best_x.copy()
            simplex = [x_nm.copy()]
            for i in range(dim):
                pert = x_nm.copy()
                pert[i] += 0.05
                simplex.append(np.clip(pert, -5, 5))

            fvals = []
            for s in simplex:
                fvals.append(func(s))
                calls += 1

            nm_iter = 0
            max_nm_iter = min(100, budget - calls - 5)
            for _ in range(max_nm_iter):
                if time.time() > deadline - 5 or calls >= budget:
                    break

                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]

                centroid = np.mean(simplex[:-1], axis=0)
                xr = np.clip(2 * centroid - simplex[-1], -5, 5)
                fr = func(xr)
                calls += 1

                if fr < fvals[0]:
                    xe = np.clip(centroid + 2 * (xr - centroid), -5, 5)
                    fe = func(xe)
                    calls += 1
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    xc = np.clip(centroid + 0.5 * (simplex[-1] - centroid), -5, 5)
                    fc = func(xc)
                    calls += 1
                    if fc < fvals[-1]:
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink simplex
                        for i in range(1, dim + 1):
                            simplex[i] = np.clip(0.5 * (simplex[i] + simplex[0]), -5, 5)
                            fvals[i] = func(simplex[i])
                            calls += 1

                if fvals[0] < best_fx:
                    best_fx = fvals[0]
                    best_x = simplex[0].copy()

                nm_iter += 1

        return best_x, best_fx
# EVOLVE-BLOCK-END
