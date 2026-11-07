"""Rewritten optimizer for BBOB f14 (Different powers) using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for BBOB f14 (Different powers)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280
        rng = np.random.default_rng(42)
        d = self.dim
        max_calls = self.budget

        best_x = None
        best_f = float('inf')
        calls_used = 0

        def try_eval(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= max_calls:
                return None
            if time.time() > deadline - 2:
                return None
            fx = func(x)
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def cma_es(x0, sigma0, max_evals, min_sigma=1e-6):
            nonlocal calls_used, best_x, best_f
            if calls_used >= max_calls:
                return

            lam = max(4, int(4 + 3 * np.log(d)))
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
            c_cov_pos = (2 + 2 * mu_eff / d) / (d + 2 + mu_eff / d)
            c_cov_neg = min(1 - c_cov_pos, 2 * (mu_eff - 2 + 1.0 / mu_eff) / (d ** 2 + d))

            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(d)
            C_inv = np.eye(d)
            pc = np.zeros(d)
            ps = np.zeros(d)
            chi_n = np.sqrt(d) * (1 - 1.0 / (4 * d) + 1.0 / (20 * d ** 2))

            # Initialize population
            pop = np.zeros((lam, d))
            pop_f = np.zeros(lam)

            for gen in range(max_evals // lam + 2):
                if calls_used >= max_calls:
                    return
                if time.time() > deadline - 3:
                    return

                # Generate population
                for i in range(lam):
                    if calls_used >= max_calls or time.time() > deadline - 3:
                        return
                    # Cholesky decomposition of C
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        L = np.eye(d)
                    z = rng.standard_normal(d)
                    x_new = x_mean + sigma * (L @ z)
                    x_new = np.clip(x_new, -5, 5)
                    fx = try_eval(x_new)
                    if fx is None:
                        return
                    pop[i] = x_new
                    pop_f[i] = fx

                # Sort by fitness
                order = np.argsort(pop_f)
                pop = pop[order]
                pop_f = pop_f[order]

                # Update best
                if pop_f[0] < best_f:
                    best_f = pop_f[0]
                    best_x = pop[0].copy()

                # Weighted mean
                x_new_mean = np.sum(weights[:, None] * pop[:mu], axis=0)

                # Update step size control vector
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (L @ (x_new_mean - x_mean))
                norm_ps = np.linalg.norm(ps)

                # Update evolution path
                pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new_mean - x_mean) / sigma

                # Update covariance matrix
                delta = 0 if norm_ps ** 2 / chi_n ** 2 < 2 * (1 + 1 / c_c) else 1
                C = (1 + c_cov_pos + c_cov_neg - c_c) * C + \
                    c_cov_pos * np.outer(pc, pc) + \
                    c_cov_neg * (1 - delta) * np.outer(ps, ps) - \
                    c_cov_neg * (1 - delta) * chi_n ** 2 * C

                # Ensure symmetry
                C = (C + C.T) / 2

                # Renormalize C to keep condition number reasonable
                eigvals = np.linalg.eigvalsh(C)
                if np.min(eigvals) < 1e-10:
                    C += 1e-10 * np.eye(d)

                # Update mean and sigma
                x_mean = x_new_mean
                sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))

                # Check termination
                if sigma < min_sigma:
                    break

                # Check if we're stuck in a corner
                if np.all(np.abs(x_mean) > 4.5):
                    # Restart locally
                    x_mean = rng.uniform(-3, 3, d)
                    sigma = 1.0
                    C = np.eye(d)
                    pc = np.zeros(d)
                    ps = np.zeros(d)

            return

        # Phase 1: Diverse initialization
        init_points = [np.zeros(d)]
        for i in range(d):
            for val in [-3, 3]:
                pt = np.zeros(d)
                pt[i] = val
                init_points.append(pt)
        for _ in range(6):
            init_points.append(rng.uniform(-4, 4, d))

        for p in init_points:
            p_clipped = np.clip(p, -5, 5)
            fx = try_eval(p_clipped)
            if fx is None:
                break

        if calls_used >= max_calls or time.time() > deadline - 20:
            return best_x, best_f

        # Phase 2: CMA-ES starting from best point
        x0 = best_x.copy()
        sigma0 = 2.0
        max_cma_evals = max_calls - calls_used
        cma_es(x0, sigma0, max_cma_evals)

        return best_x, best_f
# EVOLVE-BLOCK-END
