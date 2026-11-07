"""Optimizer for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-phase optimizer: global exploration, then CMA-ES-like local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        total_calls = 0

        def try_eval(x):
            nonlocal total_calls, best_x, best_f
            if total_calls >= budget:
                return None
            if time.time() > deadline - 3:
                return None
            x = np.clip(x, -5, 5)
            f = func(x)
            total_calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # --- Phase 1: Global exploration ---
        n_initial = 30
        structured_points = [
            np.zeros(n),
            np.ones(n) * 2,
            -np.ones(n) * 2,
            np.ones(n) * 1,
            -np.ones(n) * 1,
            np.array([3.0, -3.0, 2.0, -2.0, 1.0]),
            np.array([-2.0, 3.0, -1.0, 2.0, -3.0]),
            np.array([4.0, 4.0, -4.0, -4.0, 4.0]),
            np.array([-4.0, -4.0, 4.0, 4.0, -4.0]),
        ]
        for p in structured_points:
            if total_calls >= budget or time.time() > deadline - 5:
                break
            try_eval(p)

        for i in range(n_initial):
            if total_calls >= budget or time.time() > deadline - 5:
                break
            scale = 3.5 * (1.0 - i / max(1, n_initial)) + 0.5
            x_try = rng.normal(0, scale, n)
            try_eval(x_try)

        if best_x is None:
            return None, np.inf

        # --- Phase 2: CMA-ES-like local refinement ---
        def cma_es(x0, max_calls):
            """Simple CMA-ES variant for local optimization."""
            nonlocal total_calls, best_x, best_f
            if max_calls < 20:
                return x0, best_f

            pop_size = 4 + n  # 9 for n=5
            sigma = 0.5
            x_mean = x0.copy()
            C = np.eye(n)
            C_inv = np.eye(n)
            C_sqrt = np.eye(n)
            
            # Initialize population
            population = []
            fvals = []
            for i in range(pop_size):
                z = rng.standard_normal(n)
                x_i = x_mean + sigma * (C_sqrt @ z)
                f_i = try_eval(x_i)
                if f_i is None:
                    break
                population.append(x_i)
                fvals.append(f_i)

            if len(fvals) < 2:
                return x_mean, best_f

            # CMA-ES main loop
            mu = pop_size // 2  # number of parents
            weights = np.ones(mu)
            weights /= weights.sum()
            mu_eff = 1.0 / (weights ** 2).sum()
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            d_sigma = 1 + 2 * max(0, sqrt(mu_eff / n) - 1) + c_sigma

            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            for gen in range(max_calls // pop_size):
                if total_calls >= budget or time.time() > deadline - 5:
                    break

                # Sort by fitness
                order = np.argsort(fvals)
                population = [population[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Update mean
                x_mean_new = sum(weights[i] * population[i] for i in range(mu))

                # Update step size
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv @ (x_mean_new - x_mean)) / sigma
                h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))) < (1.4 + 2 / (n + 1)) * chi_n
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 10.0)

                # Update covariance
                if h_sigma:
                    p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean_new - x_mean) / sigma
                else:
                    p_c = (1 - c_c) * p_c

                delta_h = (1 - h_sigma) * (1 - (1 - c_c) ** 2)
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * sum(weights[i] * np.outer(population[i] - x_mean, population[i] - x_mean) / sigma ** 2 for i in range(mu))

                # Ensure positive definiteness
                C = 0.5 * (C + C.T)
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C_inv = eigvecs @ np.diag(1 / eigvals) @ eigvecs.T
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

                x_mean = x_mean_new

                # Generate new population
                population = []
                fvals = []
                for i in range(pop_size):
                    z = rng.standard_normal(n)
                    x_i = x_mean + sigma * (C_sqrt @ z)
                    f_i = try_eval(x_i)
                    if f_i is None:
                        break
                    population.append(x_i)
                    fvals.append(f_i)

                if len(fvals) < 2:
                    break

                # Update best
                if fvals[0] < best_f:
                    best_f = fvals[0]
                    best_x = population[0].copy()

            return x_mean, best_f

        # Run CMA-ES from best point
        remaining = budget - total_calls
        if remaining >= 30 and best_x is not None:
            x_opt, f_opt = cma_es(best_x.copy(), remaining)
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

        # --- Phase 3: Final polish with tiny perturbations ---
        if total_calls < budget and best_x is not None and time.time() < deadline - 3:
            n_final = min(20, budget - total_calls)
            for i in range(n_final):
                if total_calls >= budget or time.time() > deadline - 3:
                    break
                perturbation = rng.normal(0, 0.05, n)
                x_try = best_x + perturbation
                f_try = try_eval(x_try)

        return best_x, best_f
# EVOLVE-BLOCK-END
