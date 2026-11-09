"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES for BBOB f22 (Gallagher 21 peaks)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        n = self.dim
        budget = self.budget

        # CMA-ES parameters
        lam = 6  # population size
        mu = lam // 2  # number of selected parents
        weights = np.ones(mu)
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        def cma_es(x0, sigma0, remaining_budget):
            x = x0.copy()
            sigma = sigma0
            ps = np.zeros(n)
            pc = np.zeros(n)
            C = np.eye(n)
            C_sqrt = np.eye(n)
            C_inv_sqrt = np.eye(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            best_x = x0.copy()
            best_f = func(x0)
            evals = 1

            for gen in range(remaining_budget // lam + 1):
                if time.time() > deadline:
                    break
                if evals >= remaining_budget:
                    break

                # Generate population
                # Compute C^{-1/2} via eigendecomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-20)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

                # Sample lambda points
                Y = rng.standard_normal((lam, n))
                # Transform: x + sigma * C^{-1/2} @ Y
                X = x[None, :] + sigma * (C_inv_sqrt @ Y.T).T
                # Clip to box
                X = np.clip(X, -5.0, 5.0)

                # Evaluate
                Fs = np.zeros(lam)
                for i in range(lam):
                    if time.time() > deadline:
                        break
                    if evals >= remaining_budget:
                        break
                    Fs[i] = func(X[i])
                    evals += 1
                    if Fs[i] < best_f:
                        best_f = Fs[i]
                        best_x = X[i].copy()

                # Sort by fitness
                order = np.argsort(Fs)
                X_sorted = X[order]
                Fs_sorted = Fs[order]

                # Update mean
                x_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)

                # Update sigma (CSA)
                diff = (x_new - x) / (sigma * chi_n)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff
                if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (evals + 1) / (lam * c_sigma))) < (1.4 + 2 / (n + 1)) * chi_n:
                    pass  # no hsig correction needed for simplicity

                # Update pc (CM)
                hsig = 1.0  # simplified
                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * diff

                # Rank-mu update of C
                v = (X_sorted[:mu] - x[None, :]) / sigma
                C = (1 - c_1 - c_mu) * C
                C += c_1 * (np.outer(ps, ps) + (1 - hsig) * (1 - c_c) * c_c * C)
                C += c_mu * np.sum(weights[:, None, None] * v[:, :, None] * v[:, None, :], axis=0)

                # Symmetrize
                C = 0.5 * (C + C.T)

                # Update C^{-1/2} periodically (every 10 gens for efficiency)
                if gen % 10 == 0:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-20)
                    C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

                # Adapt sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

                x = x_new
                # Clip mean to stay in bounds
                x = np.clip(x, -5.0, 5.0)

                # Stagnation check: if no improvement in last 30 evals, restart
                # (handled by outer loop)

            return best_x, best_f, evals

        # Multi-restart strategy: use budget for up to 3 CMA-ES runs
        # First run: 60% of budget, second: 25%, third: 15%
        splits = [int(budget * 0.6), int(budget * 0.25), budget - int(budget * 0.6) - int(budget * 0.25)]
        splits = [max(s, 10) for s in splits]

        best_x = None
        best_f = float('inf')

        for restart in range(3):
            if time.time() > deadline:
                break
            if restart == 0:
                x0 = rng.uniform(-4, 4, n)
                sigma0 = 2.0
            else:
                # Restart from best found so far with perturbation
                x0 = best_x + 0.5 * rng.standard_normal(n)
                x0 = np.clip(x0, -4.5, 4.5)
                sigma0 = 0.5

            rx, rf, used = cma_es(x0, sigma0, splits[restart])
            if rf < best_f:
                best_f = rf
                best_x = rx

            if time.time() > deadline:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
