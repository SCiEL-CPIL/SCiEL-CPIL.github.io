"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive covariance, restarts, and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit
        rng = np.random.default_rng(12345)
        n = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        def count_call(x):
            nonlocal calls_made, best_x, best_f
            calls_made += 1
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def cmaes(func, budget, x0, sigma0, seed):
            nonlocal calls_made, best_x, best_f
            rng = np.random.default_rng(seed)
            n = self.dim
            lam = max(4, int(4 + 3 * np.log(n)))
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mueff + 2) / (n + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            x = np.array(x0, dtype=float)
            sigma = sigma0
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            C = np.eye(n)
            C_inv = np.eye(n)
            C_sqrt = np.eye(n)
            C_inv_sqrt = np.eye(n)

            def update_matrices():
                nonlocal C, C_inv, C_sqrt, C_inv_sqrt
                # Update C
                C = (1 - c_1 - c_mu) * C + c_1 * (2 - mu_c) * np.outer(p_c, p_c)
                for i in range(mu):
                    C += c_mu * weights[i] * np.outer(y_w[i], y_w[i])
                # Ensure symmetry
                C = (C + C.T) / 2
                # Compute inverse and square root
                try:
                    C_inv = np.linalg.inv(C)
                except np.linalg.LinAlgError:
                    C_inv = np.eye(n)
                # Compute sqrt and inv-sqrt via eigendecomposition
                try:
                    w, V = np.linalg.eigh(C)
                    w = np.clip(w, 1e-12, None)
                    C_sqrt = V @ np.diag(np.sqrt(w)) @ V.T
                    C_inv_sqrt = V @ np.diag(1.0 / np.sqrt(w)) @ V.T
                except np.linalg.LinAlgError:
                    C_sqrt = np.eye(n)
                    C_inv_sqrt = np.eye(n)

            # Initial evaluations
            fx = count_call(x)
            if calls_made >= budget or time.time() > deadline:
                return x, fx

            # Generate initial population
            for it in range(budget - 1):
                if calls_made >= budget or time.time() > deadline:
                    break

                # Generate offspring
                B = C_sqrt
                D = 1.0 / np.sqrt(np.clip(np.diag(B @ B.T), 1e-12, None))
                y_w = np.zeros((mu, n))
                ys = np.zeros((lam, n))
                fs = np.zeros(lam)

                for i in range(lam):
                    z = rng.standard_normal(n)
                    y = x + sigma * (B @ z)
                    y = np.clip(y, -5, 5)
                    ys[i] = y
                    fs[i] = count_call(y)
                    if calls_made >= budget or time.time() > deadline:
                        break

                if calls_made >= budget or time.time() > deadline:
                    break

                # Sort by fitness
                idx = np.argsort(fs)
                ys = ys[idx]
                fs = fs[idx]
                y_w = ys[:mu]

                # Update mean
                x_old = x.copy()
                x_new = np.sum(weights[:, None] * y_w, axis=0)
                x_new = np.clip(x_new, -5, 5)

                # Update step size control
                y_sigma = (C_inv_sqrt @ (x_new - x_old)) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * y_sigma
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (it + 1))) / chi_n < 1.4 + 2 / (n + 1)
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-12, 10.0)

                # Update evolution path
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * C_inv_sqrt @ (x_new - x_old) / sigma

                # Update covariance
                update_matrices()

                # Accept new mean
                x = x_new

                # Check for convergence
                if np.linalg.norm(x_new - x_old) < 1e-10 * sigma:
                    break

            return x, best_f if best_x is not None else fx

        # Run multiple restarts
        n_restarts = 3
        per_restart_budget = max(1, budget // n_restarts)
        for r in range(n_restarts):
            if calls_made >= budget or time.time() > deadline:
                break
            x0 = rng.uniform(-4, 4, n)
            sigma0 = 2.0 if r == 0 else 1.0
            _, _ = cmaes(func, per_restart_budget, x0, sigma0, seed=1000 + r)

        if best_x is None:
            x = np.zeros(n)
            count_call(x)
            best_x = x
            best_f = func(x)

        return best_x, best_f
# EVOLVE-BLOCK-END
