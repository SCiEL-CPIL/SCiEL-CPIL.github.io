"""Combined optimizer: Sobol global scan + CMA-ES multi-basin refinement for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Sobol global scan followed by CMA-ES refinement on top basins."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def time_ok():
            return time.time() < deadline - 5.0

        # ---- Phase 1: Sobol quasi-random global scan ----
        # Use a Sobol sequence for low-discrepancy coverage of [-5,5]^dim
        n_scan = min(40, budget // 5)
        n_scan = max(20, n_scan)

        # Generate Sobol points
        try:
            from scipy.stats.qmc import Sobol
            sampler = Sobol(d=dim, scramble=True, seed=42)
            sobol_pts = sampler.random(n_scan)
        except Exception:
            # Fallback: stratified random
            n_per_dim = max(2, int(np.ceil(n_scan ** (1.0 / dim))))
            sobol_pts = np.zeros((n_scan, dim))
            for i in range(n_scan):
                for d in range(dim):
                    bin_idx = (i + d * n_scan // n_per_dim) % n_per_dim
                    frac = (bin_idx + rng.uniform()) / n_per_dim
                    sobol_pts[i, d] = frac

        # Map to [-5, 5]
        points = sobol_pts * 10.0 - 5.0

        best_fx = float('inf')
        best_x = None
        candidates = []

        for x in points:
            if not time_ok():
                break
            fx = func(x)
            candidates.append((fx, x.copy()))
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        if not candidates or best_x is None:
            x0 = rng.uniform(-5, 5, dim)
            fx0 = func(x0)
            return x0, fx0

        candidates.sort(key=lambda t: t[0])
        remaining_budget = budget - len(candidates)

        if remaining_budget <= 0:
            return best_x, best_fx

        # ---- Phase 2: CMA-ES on top basins ----
        # Allocate 75% of remaining to best basin, 25% to second best
        budget_best = int(remaining_budget * 0.75)
        budget_second = remaining_budget - budget_best

        def run_cma_es(start_x, n_evals, seed_offset=0):
            """Run CMA-ES from start_x with n_evals budget."""
            if n_evals < 10:
                return start_x, float('inf')

            local_rng = np.random.default_rng(42 + seed_offset)

            # Initialize sigma from spread of top candidates
            if len(candidates) >= 3:
                spread = np.linalg.norm(candidates[0][1] - candidates[1][1])
                spread2 = np.linalg.norm(candidates[0][1] - candidates[2][1])
                init_sigma = max(0.3, min(0.5 * (spread + spread2), 2.0))
            else:
                init_sigma = 1.0

            mean = start_x.copy()
            C = np.eye(dim)
            sigma = init_sigma

            # CMA-ES parameters for dim=5
            lambda_pop = max(8, 4 + int(3 * np.log(dim)))
            # Ensure we can fit at least 2 generations
            if n_evals < 2 * lambda_pop:
                lambda_pop = max(4, n_evals // 3)

            mu = max(1, lambda_pop // 2)
            weights = np.zeros(mu)
            for i in range(mu):
                weights[i] = np.log(mu + 0.5) - np.log(i + 1)
            weights /= np.sum(weights)
            mu_eff = 1.0 / np.sum(weights ** 2)

            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            try:
                C_chol = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C_chol = np.eye(dim)

            local_best_fx = float('inf')
            local_best_x = start_x.copy()
            evals_used = 0
            no_improvement = 0
            stall_threshold = 3 * lambda_pop
            generation = 0

            while time_ok() and evals_used + lambda_pop <= n_evals:
                # Generate population
                offspring = []
                for i in range(lambda_pop):
                    if evals_used >= n_evals:
                        break
                    z = local_rng.standard_normal(dim)
                    y = mean + sigma * (C_chol @ z)
                    y = np.clip(y, -5.0, 5.0)
                    fy = func(y)
                    evals_used += 1
                    offspring.append((fy, y))
                    if fy < local_best_fx:
                        local_best_fx = fy
                        local_best_x = y.copy()

                if len(offspring) < 2:
                    break

                offspring.sort(key=lambda t: t[0])

                # Track global improvement
                if offspring[0][0] < best_fx - 1e-10:
                    best_fx = offspring[0][0]
                    best_x = offspring[0][1].copy()
                    no_improvement = 0
                else:
                    no_improvement += 1

                # Update mean
                m_actual = min(mu, len(offspring))
                new_mean = np.zeros(dim)
                for i in range(m_actual):
                    new_mean += weights[i] * offspring[i][1]

                # Compute C^{-1/2}
                try:
                    C_sqrt = np.linalg.cholesky(C + 1e-15 * np.eye(dim))
                    C_inv_sqrt = np.linalg.inv(C_sqrt)
                except np.linalg.LinAlgError:
                    C_inv_sqrt = np.eye(dim)

                # Update path variables
                delta_mean = new_mean - mean
                p_sigma = (1 - c_sigma) * p_sigma + \
                    np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ delta_mean / sigma
                p_c = (1 - c_c) * p_c + \
                    np.sqrt(c_c * (2 - c_c) * mu_eff) * delta_mean / sigma

                # Update sigma
                expected_norm = np.sqrt(mu_eff) * (1 + 1 / (dim + 1))
                threshold = (1.4 + 2 / (dim + 1)) * expected_norm
                denom = np.sqrt(1 - (1 - c_sigma) ** (2 * (generation + 1)))
                if denom < 1e-10:
                    denom = 1.0
                h_sigma = 1.0 if np.linalg.norm(p_sigma) / denom < threshold else 0.0

                sigma *= np.exp(c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 10.0)

                # Update covariance matrix
                delta_h = h_sigma * (1 / c_sigma - 1)
                rank_one = np.outer(p_c, p_c)
                rank_mu = np.zeros((dim, dim))
                for i in range(m_actual):
                    diff = offspring[i][1] - mean
                    rank_mu += weights[i] * np.outer(diff, diff)

                C = (1 + c_1 * (delta_h - 1) + c_mu) * C + c_1 * rank_one + c_mu * rank_mu
                C = 0.5 * (C + C.T)

                try:
                    C_chol = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                except np.linalg.LinAlgError:
                    C_chol = np.eye(dim)

                mean = new_mean
                generation += 1

                # Stagnation: restart locally
                if no_improvement > stall_threshold:
                    mean = local_best_x.copy()
                    sigma = max(0.2, sigma * 0.5)
                    C = np.eye(dim)
                    C_chol = np.eye(dim)
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
                    no_improvement = 0
                    generation = 0

            return local_best_x, local_best_fx

        # Run CMA-ES on best basin
        x_best, f_best = run_cma_es(best_x, budget_best, seed_offset=0)
        if f_best < best_fx:
            best_x = x_best.copy()
            best_fx = f_best

        # Run CMA-ES on second-best basin if budget allows
        if len(candidates) > 1 and budget_second >= 8:
            x_second, f_second = candidates[1]
            x_s, f_s = run_cma_es(x_second, budget_second, seed_offset=100)
            if f_s < best_fx:
                best_x = x_s.copy()
                best_fx = f_s

        # ---- Phase 3: Final local polish with remaining budget ----
        if time_ok():
            used = budget - (budget - len(candidates) - budget_best - (budget_second if len(candidates) > 1 and budget_second >= 8 else 0))
            # Simpler: just check remaining calls
            remaining = budget - (len(candidates) + budget_best + (budget_second if len(candidates) > 1 and budget_second >= 8 else 0))
            if remaining > 3:
                polish_budget = min(remaining, 15)
                x = best_x.copy()
                fx = best_fx
                step = 0.05
                for _ in range(polish_budget):
                    if not time_ok():
                        break
                    y = x + step * rng.standard_normal(dim)
                    y = np.clip(y, -5.0, 5.0)
                    fy = func(y)
                    if fy < fx:
                        x = y
                        fx = fy
                        if fx < best_fx:
                            best_fx = fx
                            best_x = x.copy()
                    else:
                        step *= 0.7

        return best_x, best_fx
# EVOLVE-BLOCK-END
