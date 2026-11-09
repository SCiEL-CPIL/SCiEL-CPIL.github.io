"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts for BBOB f13."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280.0  # 300s limit, 20s safety margin
        total_budget = self.budget
        dim = self.dim

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x, best_f = x.copy(), f

        # Phase 1: CMA-ES with adaptive restarts
        # We use a single CMA-ES run with restarts based on stagnation

        def run_cmaes(x0, sigma0, budget_left, rng):
            n = dim
            x = np.array(x0, dtype=float)
            f = func(x)
            update_best(x, f)

            sigma = sigma0
            C = np.eye(n)
            # Precompute Cholesky of C for sampling
            try:
                C_chol = np.linalg.cholesky(C)
            except:
                C_chol = np.eye(n)

            # CMA-ES parameters
            mu = max(1, n // 2)
            lambda_ = 2 * mu
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
            c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
            c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma

            budget_used = 1
            stall_count = 0
            best_in_run = f

            for _ in range(budget_left - 1):
                if time.time() > deadline:
                    break
                if budget_used >= budget_left:
                    break

                # Generate offspring
                ys = []
                for _ in range(lambda_):
                    z = rng.standard_normal(n)
                    y = x + sigma * (C_chol @ z)
                    y = np.clip(y, -5, 5)
                    ys.append(y)

                # Evaluate
                fs = []
                for y in ys:
                    fy = func(y)
                    fs.append(fy)
                    budget_used += 1
                    if fy < best_in_run:
                        best_in_run = fy
                        update_best(y, fy)
                    if time.time() > deadline:
                        break

                if len(fs) < lambda_:
                    break

                fs = np.array(fs)
                ys = np.array(ys)

                # Sort
                idx = np.argsort(fs)
                ys = ys[idx]
                fs = fs[idx]

                # Update mean
                x_new = np.sum(weights[:, None] * ys[:mu], axis=0)

                # Update sigma
                y_sigma = (x_new - x) / (sigma * chi_n)
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
                h_sig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (budget_used + 1))) < (1.4 + 2 / (n + 1)) * chi_n
                if h_sig:
                    p_c = (1 - c_c) * p_c + c_c * y_sigma
                    C = (1 - c_1 - c_mu) * C + c_1 * (p_c @ p_c - C) + c_mu * np.sum(weights[:, None, None] * (ys[:mu] - x)[:, None, :] * (ys[:mu] - x)[:, None, :].T, axis=0) / (sigma ** 2)
                    # Symmetrize
                    C = (C + C.T) / 2
                    # Regularize
                    C += 1e-12 * np.eye(n)
                    try:
                        C_chol = np.linalg.cholesky(C)
                    except:
                        C_chol = np.eye(n)
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

                x = x_new
                f = func(x)
                budget_used += 1
                if f < best_in_run:
                    best_in_run = f
                    update_best(x, f)

                # Stagnation check
                if f >= best_in_run + 1e-10:
                    stall_count += 1
                else:
                    stall_count = 0

                if stall_count > 30:
                    break

            return x, f, budget_used

        # Initial points for restarts
        rng = np.random.default_rng(42)
        x0 = np.zeros(dim)
        sigma0 = 2.0

        remaining = total_budget
        restarts = 0
        max_restarts = 3

        while remaining > 50 and restarts <= max_restarts and time.time() < deadline:
            x, f, used = run_cmaes(x0, sigma0, min(remaining, 200), rng)
            remaining -= used
            restarts += 1

            if time.time() > deadline:
                break

            # If we haven't found good enough, restart from best found
            if best_f < np.inf:
                x0 = best_x.copy()
                sigma0 = 1.0
            else:
                x0 = rng.uniform(-5, 5, dim)
                sigma0 = 2.0

        # Use remaining budget for local refinement around best point
        if remaining > 0 and best_x is not None and time.time() < deadline:
            x = best_x.copy()
            f = best_f
            sigma = 0.5
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                if fy < f:
                    x, f = y, fy
                    sigma *= 1.2
                else:
                    sigma *= 0.9
                update_best(x, f)

        return best_x, best_f
# EVOLVE-BLOCK-END
