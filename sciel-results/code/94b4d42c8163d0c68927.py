"""CMA-ES optimizer for BBOB f18 (Schaffers F7 ill-conditioned)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts for ill-conditioned black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 300s limit
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

        rng = np.random.default_rng(42)

        # Phase 1: Random sampling to find a good starting point
        n_random = min(20, budget)
        for _ in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-4, 4, dim)
            call_func(x)

        # Phase 2: CMA-ES
        def run_cmaes(x0, sigma0, remaining_budget):
            nonlocal calls, best_x, best_f, deadline
            if calls >= budget or time.time() > deadline:
                return

            # CMA-ES parameters
            n = dim
            lam = max(4, int(4 + 3 * np.log(n)))  # population size
            mu = lam // 2  # number of weights
            # Weights (log-sum normalization)
            w_raw = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            w = w_raw / w_raw.sum()
            w = np.concatenate([w, np.zeros(lam - mu)])
            mueff = 1.0 / np.sum(w[:mu] ** 2)

            # CMA-ES state
            m = x0.copy()  # mean
            sigma = sigma0
            C = np.eye(n)  # covariance matrix
            C_inv = np.eye(n)
            pc = np.zeros(n)  # evolution path for covariance
            ps = np.zeros(n)  # evolution path for step-size

            # Strategy parameters
            c_sigma = (mueff + 2) / (n + mueff + 5)
            d_sigma = 1 + 2 * max(0, (mueff - 1) / (n + 1) - 1) + c_sigma
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

            # Initial Cholesky factor
            C_sqrt = np.eye(n)

            while calls < budget:
                if time.time() > deadline:
                    break

                # Generate population
                offspring = []
                for i in range(lam):
                    z = rng.standard_normal(n)
                    y = m + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5.0, 5.0)
                    offspring.append((y, call_func(y)))

                # Sort by fitness
                offspring.sort(key=lambda t: t[1])

                # Update mean
                m_new = np.sum(w[:, np.newaxis] * np.array([o[0] for o in offspring]), axis=0)

                # Update evolution paths
                diff = (m_new - m) / sigma
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * diff
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (calls + 1))) <= \
                       (1 + 2 / (n + 1)) * np.sqrt(1 - (1 - c_sigma) ** (calls + 1))
                if hsig:
                    pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * diff

                # Update covariance matrix
                delta_hsig = (1 - hsig) * c_c * (2 - c_c)
                delta_hsig = delta_hsig if hsig else 0
                C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C + \
                    c_1 * np.outer(pc, pc) + \
                    c_mu * np.sum(w[:mu] * np.outer(np.array([offspring[i][0] for i in range(mu)]) - m,
                                                     np.array([offspring[i][0] for i in range(mu)]) - m), axis=0)

                # Ensure C is symmetric and positive definite
                C = (C + C.T) / 2
                try:
                    C_inv = np.linalg.inv(C)
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = C + 1e-10 * np.eye(n)
                    C_inv = np.linalg.inv(C)
                    C_sqrt = np.linalg.cholesky(C)

                # Update sigma (CSA)
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (calls + 1)) - np.sqrt(n)))

                # Check for convergence
                if sigma < 1e-8:
                    break

                # Update mean
                m = m_new

            # Return best found
            return m, sigma

        # Run CMA-ES from the best random point
        x0 = best_x.copy()
        sigma0 = 1.0
        run_cmaes(x0, sigma0, budget - calls)

        # Phase 3: If we still have budget, try restarts from different points
        restarts = 0
        max_restarts = 3
        while calls < budget and restarts < max_restarts and time.time() < deadline:
            # Random restart
            x_r = rng.uniform(-4, 4, dim)
            f_r = call_func(x_r)

            if f_r < best_f:
                best_x = x_r.copy()
                best_f = f_r

            # Run CMA-ES from this point with smaller sigma
            x0 = x_r.copy()
            sigma0 = 0.5
            run_cmaes(x0, sigma0, budget - calls)
            restarts += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
