"""Optimizer for BBOB f24 (Lunacek bi-Rastrigin) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        n = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        eval_count = 0

        def safe_func(x):
            nonlocal best_x, best_f, eval_count
            eval_count += 1
            try:
                fx = float(func(x))
            except Exception:
                fx = np.inf
            if np.isnan(fx) or np.isinf(fx):
                fx = np.inf
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def time_exceeded():
            return time.time() > deadline

        def enough_budget():
            return eval_count >= budget

        rng = np.random.default_rng(42)

        # --- Phase 1: Diverse random sampling ---
        num_initial = 20
        for _ in range(num_initial):
            if time_exceeded() or enough_budget():
                break
            x = rng.uniform(-4.5, 4.5, n)
            safe_func(x)

        # --- Phase 2: CMA-ES from best point ---
        if best_x is not None and not time_exceeded() and not enough_budget():
            lam = 4 + int(3 * math.log(n))
            mu = lam // 2
            weights = np.log(lam - 0.5) - np.log(np.arange(1, lam))
            weights /= weights.sum()
            mu_eff = 1 / (weights ** 2).sum()

            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / n) / ((n + 2) ** 2 + mu_eff))
            c_c = (4 + math.log(n)) / (4 * n + math.log(n))
            c_s = (n + 2) / (n + 4 + 2 * n / (n + 1))
            chi_n = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))

            x = best_x.copy().astype(float)
            sigma = 1.0
            C = np.eye(n)
            p_c = np.zeros(n)
            p_s = np.zeros(n)

            def mat_sqrt_inv(A):
                eigvals, eigvecs = np.linalg.eigh(A)
                eigvals = np.clip(eigvals, 1e-12, None)
                return eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T

            C_inv_sqrt = mat_sqrt_inv(C)

            it = 0
            while not time_exceeded() and not enough_budget():
                # Generate population
                pop = np.array([x + sigma * (C_inv_sqrt @ rng.standard_normal(n)) for _ in range(lam)])
                pop = np.clip(pop, -5, 5)
                fvals = np.array([safe_func(p) for p in pop])

                if time_exceeded() or enough_budget():
                    break

                indices = np.argsort(fvals)
                y_w = pop[indices[:mu]]
                w = weights[:mu]

                x_old = x.copy()
                x = x + sigma * (w @ (y_w - x).T)

                # Convergence check
                if np.linalg.norm(x - x_old) / max(sigma, 1e-10) < 1e-7:
                    break

                y = (y_w - x_old) / sigma
                y_mean = w @ y.T
                p_s = (1 - c_s) * p_s + math.sqrt(c_s * (2 - c_s) * mu_eff) * y_mean
                hsig = (np.linalg.norm(p_s) / math.sqrt(1 - (1 - c_s) ** (it + 1)) / chi_n - 1) < 1.4 + 2 / (n + 1)
                p_c = (1 - c_c) * p_c + hsig * c_c * (x - x_old) / sigma
                C = (1 + c_1 * (2 - mu_eff) * hsig - c_1 - c_mu) * C \
                    + c_1 * (2 - mu_eff) * np.outer(p_c, p_c) / (1 - (1 - c_c) ** (it + 1)) \
                    + c_mu * (w @ (y @ (y * w[:, None])).T)

                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C_inv_sqrt = mat_sqrt_inv(C)

                # Simple step size adaptation
                if fvals[indices[0]] < fvals[indices[-1]]:
                    sigma *= 1.0
                else:
                    sigma *= 0.9

                if sigma < 1e-10:
                    break

                it += 1

        # --- Phase 3: Local (1+1)-ES refinement ---
        if best_x is not None and not time_exceeded() and not enough_budget():
            x = best_x.copy()
            fx = best_f
            sigma = 0.3
            successes = 0
            total = 0
            while not time_exceeded() and not enough_budget():
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = safe_func(y)
                total += 1
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                if total >= 10:
                    rate = successes / total
                    if rate > 1 / 5:
                        sigma *= 1.4
                    elif rate < 1 / 5:
                        sigma *= 0.75
                    successes = 0
                    total = 0
                    if sigma < 1e-8:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
