"""CMA-ES with population-based search for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive step-size and covariance for ill-conditioned landscapes."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls = 0

        def call_func(x):
            nonlocal calls, best_x, best_f
            x = np.asarray(x, dtype=np.float64)
            x = np.where(np.isfinite(x), x, 0.0)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(12345)

        # Phase 1: Random sampling
        n_random = min(30, budget)
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4, 4, dim)
            call_func(x)

        # Phase 2: CMA-ES-like optimization
        x = best_x.copy()
        sigma = 1.0
        C = np.eye(dim)  # Covariance matrix
        Cs = np.zeros(dim)  # Evolution path for step-size
        C_path = np.zeros(dim)  # Evolution path for covariance
        H = 0.0  # Hessian-like term for covariance update

        # CMA-ES parameters
        lambda_pop = max(4, 4 + int(3 * np.log(dim)))  # Population size
        mu = lambda_pop // 2  # Number of parents
        # Weights: log-linear
        w = np.log(np.arange(mu, 0, -1)) - np.log(np.arange(mu, 0, -1)).mean()
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # Track best for restart
        best_in_run = best_f
        stagnation = 0
        max_stagnation = 50

        while calls < budget and time.time() < deadline:
            if time.time() > deadline:
                break

            # Generate population
            ys = []
            fs = []
            for i in range(lambda_pop):
                if calls >= budget or time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                # Sample from N(0, C) using Cholesky
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                y = x + sigma * (L @ z)
                y = np.clip(y, -5.0, 5.0)
                fy = call_func(y)
                ys.append(y)
                fs.append(fy)

            if len(ys) < 2:
                break

            # Sort by fitness
            order = np.argsort(fs)
            ys_sorted = [ys[i] for i in order]
            fs_sorted = [fs[i] for i in order]

            # Update best
            if fs_sorted[0] < best_in_run:
                best_in_run = fs_sorted[0]
                stagnation = 0
            else:
                stagnation += 1

            # Update mean using weighted average of top mu
            x_new = np.zeros(dim)
            for i in range(min(mu, len(ys_sorted))):
                x_new += w[i] * ys_sorted[i]
            x = x_new

            # Update evolution path for step-size
            y_mean = (x_new - (x - sigma * (C @ Cs))) / sigma if sigma > 0 else np.zeros(dim)
            Cs = (1 - c_sigma) * Cs + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(Cs) / np.sqrt(dim) - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Update covariance
            # Compute rank-1 and rank-mu updates
            delta = np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
            H += (1 - c_c) * H + delta * Cs  # Simplified
            # Actually, let's use a simpler covariance update
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(Cs, Cs) + (1 - c_c) * C)
            for i in range(min(mu, len(ys_sorted))):
                zi = (ys_sorted[i] - (x_new - sigma * (C @ Cs))) / sigma if sigma > 0 else np.zeros(dim)
                C += c_mu * w[i] * np.outer(zi, zi)

            # Ensure C is positive definite
            C = (C + C.T) / 2
            C += 1e-12 * np.eye(dim)

            # Check for restart
            if stagnation > max_stagnation or sigma < 1e-6:
                # Restart from a random point near current best
                new_x = best_x + 0.5 * rng.standard_normal(dim)
                new_x = np.clip(new_x, -4, 4)
                new_f = call_func(new_x)
                if new_f < best_f:
                    best_f = new_f
                    best_x = new_x.copy()
                    x = new_x.copy()
                else:
                    x = best_x.copy()
                sigma = 1.0
                C = np.eye(dim)
                Cs = np.zeros(dim)
                H = 0.0
                stagnation = 0
                best_in_run = best_f

        return best_x, best_f
# EVOLVE-BLOCK-END
