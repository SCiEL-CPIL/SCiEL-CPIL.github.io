"""Rewritten optimizer: CMA-ES with restarts and time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multiple restarts, time-guarded."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Global best
        best_x = np.full(dim, 0.0)
        best_f = np.inf

        def evaluate(x):
            return float(func(np.clip(x, -5, 5)))

        # Number of restarts: we split budget among them
        # Use up to 5 restarts with decreasing budgets
        restarts = [
            (int(budget * 0.40), 3.0),   # large sigma, broad search
            (int(budget * 0.25), 2.0),
            (int(budget * 0.15), 1.5),
            (int(budget * 0.10), 1.0),
            (int(budget * 0.10), 0.5),
        ]

        calls_used = 0

        for restart_idx, (restart_budget, sigma0) in enumerate(restarts):
            if calls_used >= budget:
                break
            if time.time() > deadline:
                break

            # Initialize: for first restart, sample broadly; for later, near best
            if restart_idx == 0:
                x = rng.uniform(-5, 5, dim)
            else:
                # Perturb around current best
                x = best_x + rng.standard_normal(dim) * 0.5
                x = np.clip(x, -5, 5)

            fx = evaluate(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # CMA-ES parameters
            n = dim
            mu = max(1, n // 2)  # number of parents
            lambda_ = 2 * n  # population size
            # Weights
            weights = np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            # State
            sigma = sigma0
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            C = np.eye(n)
            # Precompute
            B = np.eye(n)
            D = np.ones(n)
            # Eigen-decomp helper
            def update_C(C, B, D, p_sigma, p_c, c_sigma, c_c, c_1, c_mu, weights, y, x, sigma, n, mu, chi_n):
                # Update step size
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * B.T @ (y / sigma)
                # Update evolution path
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2))) / np.sqrt(1 - (1 - c_sigma) ** (2 * mu_eff)) if np.linalg.norm(p_sigma) > 0 else 0
                h_sigma = min(1.0, h_sigma)
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * B.T @ (y / sigma)
                # Update covariance
                C = (1 + c_1 * c_mu - c_1) * C + c_1 * (np.outer(p_c, p_c) - (1 - c_c) * C) + c_mu * sum(w * np.outer(yi, yi) for w, yi in zip(weights, y))
                # Re-orthogonalize
                B, D2 = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D2, 1e-12))
                B = B / D
                return p_sigma, p_c, C, B, D

            # Main loop for this restart
            for iteration in range(restart_budget):
                if calls_used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Generate population
                ys = []
                xs = []
                for i in range(lambda_):
                    z = rng.standard_normal(n)
                    y = B @ (D * z)
                    xi = x + sigma * y
                    xi = np.clip(xi, -5, 5)
                    fi = evaluate(xi)
                    calls_used += 1
                    ys.append(xi - x)
                    xs.append((fi, xi))

                # Sort by fitness
                xs.sort(key=lambda t: t[0])
                # Take mu best
                y_sorted = [ys[i] for i in range(mu)]
                f_best = xs[0][0]
                x_best = xs[0][1]

                if f_best < best_f:
                    best_f = f_best
                    best_x = x_best.copy()

                # Move mean to weighted average
                x = sum(w * xs[i][1] for w, i in zip(weights, range(mu)))

                # Update CMA state
                p_sigma, p_c, C, B, D = update_C(C, B, D, p_sigma, p_c, c_sigma, c_c, c_1, c_mu, weights, y_sorted, x, sigma, n, mu, chi_n)

                # Adapt sigma
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
