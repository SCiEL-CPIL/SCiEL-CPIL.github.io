"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES with deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = np.full(dim, np.nan)
        best_f = np.inf
        calls = 0

        def time_exceeded():
            return time.time() >= deadline

        # Phase 1: Brief random exploration to find a good starting point
        n_explore = min(20, budget // 5)
        for i in range(n_explore):
            if time_exceeded() or calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES optimization
        if calls < budget and not time_exceeded():
            x = best_x.copy()
            f = best_f

            # Initialize CMA-ES parameters
            sigma = 1.0  # Initial step size
            C = np.eye(dim)  # Covariance matrix
            Cinv = np.eye(dim)  # Inverse covariance
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            # CMA-ES hyperparameters
            mu = max(1, dim // 2)  # Number of parents
            w = np.zeros(mu)
            w_sum = 0.0
            for i in range(mu):
                w[i] = np.log(dim / 2 + 0.5) - np.log(i + 1)
            w_sum = np.sum(w)
            w /= w_sum
            mu_eff = 1.0 / np.sum(w**2)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))

            # Precompute Cholesky decomposition
            try:
                B = np.linalg.cholesky(C)
                D = np.sqrt(np.diag(C))
                Binv = np.linalg.inv(B)
            except np.linalg.LinAlgError:
                B = np.eye(dim)
                D = np.ones(dim)
                Binv = np.eye(dim)

            # CMA-ES main loop
            while calls < budget and not time_exceeded():
                # Generate lambda offspring
                lam = max(2 * mu, 4)
                z = rng.standard_normal((lam, dim))
                y = z @ B.T  # Transform using Cholesky
                x_new = x + sigma * y
                x_new = np.clip(x_new, -5, 5, axis=1)

                # Evaluate offspring
                f_new = []
                for i in range(lam):
                    if time_exceeded() or calls >= budget:
                        break
                    fi = func(x_new[i])
                    calls += 1
                    f_new.append(fi)
                    if fi < best_f:
                        best_f = fi
                        best_x = x_new[i].copy()

                if len(f_new) < lam:
                    break

                f_new = np.array(f_new)
                idx = np.argsort(f_new)
                x_new = x_new[idx]
                f_new = f_new[idx]

                # Update mean
                x_old = x.copy()
                x = np.sum(w[:, None] * x_new[:mu], axis=0)
                x = np.clip(x, -5, 5)

                # Update sigma
                y_p = (x - x_old) / (sigma * np.linalg.norm(B.T @ np.ones(dim)))
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sign(
                    np.linalg.norm(y_p) / np.sqrt(mu_eff) - 1
                ) * np.linalg.norm(y_p)
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                sigma = max(sigma, 1e-10)

                # Update covariance
                y_c = (x - x_old) / sigma
                p_c = (1 - c_c) * p_c + c_c * y_c
                C = ((1 + c_1 * (2 - c_1) - c_mu) * C +
                     c_1 * np.outer(y_c, y_c) +
                     c_mu * np.sum(w[:, None] * np.outer(y_new, y_new) for y_new in (x_new[:mu] - x_old) / sigma))
                C = (C + C.T) / 2

                # Update Cholesky
                try:
                    B = np.linalg.cholesky(C)
                    D = np.sqrt(np.diag(C))
                    Binv = np.linalg.inv(B)
                except np.linalg.LinAlgError:
                    B = np.eye(dim)
                    D = np.ones(dim)
                    Binv = np.eye(dim)

                # Restart if sigma is too small or too large
                if sigma < 1e-8 or sigma > 100:
                    break

            # Phase 3: Local refinement with (1+1)-ES around best found
            if calls < budget and not time_exceeded() and np.isfinite(best_f):
                x = best_x.copy()
                f = best_f
                sigma_local = max(0.1, sigma * 0.5) if sigma < 1 else 0.5
                for _ in range(budget - calls):
                    if time_exceeded():
                        break
                    y = x + sigma_local * rng.standard_normal(dim)
                    y = np.clip(y, -5, 5)
                    fy = func(y)
                    calls += 1
                    if fy < f:
                        x, f = y, fy
                        sigma_local *= 1.1
                        if f < best_f:
                            best_f = f
                            best_x = x.copy()
                    else:
                        sigma_local *= 0.9
                    if sigma_local < 1e-8:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
