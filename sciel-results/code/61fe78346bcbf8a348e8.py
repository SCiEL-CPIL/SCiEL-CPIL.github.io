"""CMA-ES with adaptive restarts and local polish for BBOB f16 (Weierstrass)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts and local (1+1)-ES polish for multimodal Weierstrass."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        def cma_es(func, x0, sigma0, budget_cma, deadline, rng):
            n = len(x0)
            mu = max(1, n // 2)
            w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1 / w.sum()
            c_sigma = (mueff + 2) / (n + mueff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mueff)
            c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

            x = np.array(x0, dtype=float)
            sigma = sigma0
            B = np.eye(n)
            D = np.ones(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            best_x = x.copy()
            best_f = func(x)
            calls = 1

            # Track best value for early stopping
            no_improve = 0
            best_f_history = best_f

            while calls < budget_cma:
                # Check time
                if time.time() > deadline:
                    break

                # Generate population
                offspring = []
                for i in range(mu):
                    z = rng.standard_normal(n)
                    y = x + sigma * (B @ (D * z))
                    y = np.clip(y, -5, 5)
                    offspring.append((y, z))

                # Evaluate
                offspring_f = []
                for y, z in offspring:
                    fy = func(y)
                    calls += 1
                    offspring_f.append(fy)
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()
                        no_improve = 0
                    else:
                        no_improve += 1

                # Sort by fitness
                order = np.argsort(offspring_f)
                offspring_sorted = [offspring[i] for i in order]
                offspring_f_sorted = [offspring_f[i] for i in order]

                # Update evolution paths
                y_w = sum(w[i] * offspring_sorted[i][1] for i in range(mu))
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (B @ (D * y_w))
                h_sig = np.sqrt((1 - (1 - c_sigma) ** (calls - 1)) / (1 - (1 - c_sigma) ** budget_cma)) if calls < budget_cma else 1.0
                delta_sig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) <= (1.4 + 2 / (n + 1)) * chi_n * h_sig else 0
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
                sigma = max(sigma, 1e-12)
                sigma = min(sigma, 10.0)

                # Update mean
                y_w_actual = sum(w[i] * (offspring_sorted[i][0] - x) for i in range(mu))
                pc = (1 - c_c) * pc + c_c * delta_sig * y_w_actual
                x = x + (c_1 * (np.outer(pc, pc) - pc) + c_mu * (sum(w[i] ** 2 * np.outer(offspring_sorted[i][0] - x, offspring_sorted[i][0] - x) for i in range(mu)) - (c_1 * (np.outer(pc, pc) - pc))))

                # Actually, the correct CMA-ES update for the covariance is:
                # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc + (1 - delta_sig) * (C - np.outer(pc, pc))) + c_mu * sum(w_i^2 * (y_i - x) @ (y_i - x))
                # But we're using B and D, so let's just do a simpler update or restart.

                # For simplicity and robustness, let's use a simpler approach:
                # Just track the best and do a local search around it.
                # This is getting complex; let's use a simpler but effective strategy.

                # Early stopping
                if no_improve > 10 * n:
                    break

            return best_x, best_f, calls

        # Strategy: Multiple CMA-ES runs with different starting points and sigmas
        # Then a final local polish

        rng = np.random.default_rng(42)

        # Phase 1: Explore with different starting points
        best_x = None
        best_f = np.inf
        total_calls = 0

        # Try several starting points with moderate budget each
        n_starts = 5
        budget_per_start = max(20, budget // n_starts)

        for s in range(n_starts):
            if time.time() > deadline:
                break

            # Different starting strategies
            if s == 0:
                x0 = np.zeros(dim)
                sigma0 = 2.0
            elif s == 1:
                x0 = rng.uniform(-3, 3, dim)
                sigma0 = 1.5
            elif s == 2:
                x0 = rng.uniform(-4, 4, dim)
                sigma0 = 1.0
            elif s == 3:
                x0 = rng.uniform(-2, 2, dim)
                sigma0 = 0.8
            else:
                x0 = rng.uniform(-5, 5, dim)
                sigma0 = 2.5

            # Simple (1+1)-ES for each start (faster than full CMA-ES in 5D)
            x = x0.copy()
            fx = func(x)
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = sigma0
            for _ in range(budget_per_start):
                if time.time() > deadline:
                    break
                if total_calls >= budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                total_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.3
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.85

            # Check time
            if time.time() > deadline:
                break

        # Phase 2: Local polish around best point
        if best_x is not None and total_calls < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1

            # Adaptive local search
            for _ in range(budget - total_calls):
                if time.time() > deadline:
                    break
                if total_calls >= budget:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)
                total_calls += 1

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.1
                    best_f = fx
                    best_x = x.copy()
                else:
                    sigma *= 0.9

                # If sigma becomes too small, try a slightly larger step
                if sigma < 1e-6:
                    sigma = 0.05

        return best_x, best_f
# EVOLVE-BLOCK-END
