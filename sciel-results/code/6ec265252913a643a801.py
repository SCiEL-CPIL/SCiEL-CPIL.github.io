"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts and time guard for BBOB f20 (Schwefel)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf

        def time_check():
            return time.time() < deadline

        # CMA-ES implementation
        def cma_es(func, x0, sigma0, max_iter, time_check):
            n = dim
            # Standard CMA-ES parameters
            mu = max(1, n // 2)
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mueff + 2) / (n + mueff + 5)
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

            x = np.array(x0, dtype=float)
            sigma = sigma0
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            C = np.eye(n)
            C_sqrt_inv = np.eye(n)

            # Precompute for Cholesky
            def update_C_sqrt_inv(C):
                # Use eigendecomposition for stability
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                return C_sqrt_inv

            C_sqrt_inv = update_C_sqrt_inv(C)

            fx = func(x)
            best_x_local = x.copy()
            best_f_local = fx
            archive = [(x.copy(), fx)]

            for it in range(max_iter):
                if not time_check():
                    break

                # Sample offspring
                z = np.random.randn(mu, n)
                # Whitening: y = C_sqrt_inv @ z
                y = z @ C_sqrt_inv.T
                pops = x[None, :] + sigma * y
                fits = []
                for i in range(mu):
                    pops[i] = np.clip(pops[i], -5, 5)
                    fits.append(func(pops[i]))
                    if fits[-1] < best_f_local:
                        best_f_local = fits[-1]
                        best_x_local = pops[i].copy()

                # Sort by fitness
                order = np.argsort(fits)
                pops_sorted = pops[order]
                fits_sorted = np.array(fits)[order]

                # Update mean
                x_new = np.sum(weights[:, None] * pops_sorted, axis=0)

                # Update sigma
                z_mean = np.sum(weights[:, None] * y[order], axis=0)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * z_mean
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1)))) < (1.4 + 2 / (n + 1)) * np.sqrt((mueff + 1) / (n + 1))
                sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mueff) - 1))

                # Update evolution path
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) / sigma * (x_new - x)

                # Update covariance
                delta_h = (1 - h_sigma) * c_c * (2 - c_c)
                rank_one = np.outer(p_c, p_c)
                rank_mu = np.sum(weights[:, None, None] * np.outer(pops_sorted, pops_sorted), axis=0)
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * (rank_one + delta_h * np.eye(n)) + c_mu * rank_mu

                # Ensure symmetry and positive definiteness
                C = (C + C.T) / 2
                C_sqrt_inv = update_C_sqrt_inv(C)

                x = x_new
                archive.append((x.copy(), fx))

            return best_x_local, best_f_local

        # Restart strategy
        n_restarts = 3
        calls_per_restart = budget // n_restarts

        for r in range(n_restarts):
            if not time_check():
                break
            if budget - (r + 1) * calls_per_restart < 10:
                calls_per_restart = max(10, (budget - r * calls_per_restart) // (n_restarts - r))

            # Initial point: center or random
            if r == 0:
                x0 = np.zeros(dim)
            else:
                # Perturb around previous best
                if best_x is not None:
                    x0 = best_x + 2.0 * np.random.randn(dim)
                else:
                    x0 = np.random.uniform(-5, 5, dim)
            x0 = np.clip(x0, -5, 5)

            # Try CMA-ES
            try:
                x_best, f_best = cma_es(func, x0, 2.0, calls_per_restart, time_check)
                if f_best < best_f:
                    best_f = f_best
                    best_x = x_best.copy()
            except Exception:
                pass

            if not time_check():
                break

        # Final fallback: if no valid result, do a simple random search
        if best_x is None:
            x = np.zeros(dim)
            best_f = func(x)
            best_x = x.copy()
            for _ in range(min(50, budget)):
                if not time_check():
                    break
                y = np.random.uniform(-5, 5, dim)
                fy = func(y)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
