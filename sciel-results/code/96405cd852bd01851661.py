import time
import numpy as np


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        total_budget = self.budget
        t0 = time.perf_counter()
        deadline = t0 + 280.0  # safety margin before 300s

        best_x = None
        best_f = np.inf
        call_count = 0

        def remaining():
            nonlocal call_count
            return total_budget - call_count

        def time_ok():
            return time.perf_counter() < deadline

        # Restart schedule: a few short exploratory runs, then one long refinement
        # For 500 budget: e.g. 3 exploratory runs of ~90, then 1 refinement of ~220
        # We'll split dynamically.
        n_restarts = 3
        # First n_restarts-1 runs are exploratory (shorter), last is refinement (longer)
        base = total_budget // (n_restarts + 1)
        # Give the last run more budget for refinement
        allocs = [base] * (n_restarts - 1) + [total_budget - base * (n_restarts - 1)]

        seed = 42
        rng = np.random.default_rng(seed)

        for run_idx, alloc in enumerate(allocs):
            if not time_ok():
                break
            if remaining() <= 0:
                break

            # Initialize: for first run, random; for later runs, near best if available
            if run_idx == 0 or best_x is None:
                x = rng.uniform(-5, 5, dim)
            else:
                # Perturb the best point for the refinement run
                x = np.clip(best_x + rng.standard_normal(dim) * 0.5, -5, 5)

            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            # CMA-ES parameters
            sigma = 2.0 if run_idx < n_restarts - 1 else 1.0
            # Population size for CMA-ES: 4 + floor(3 ln dim)
            lam = max(4, 4 + int(3 * np.log(dim)))
            # But we want to use the full alloc budget, so lam is small for many iterations
            # Use a small population to get more iterations
            lam = min(lam, max(4, alloc // 50))  # at least 4, at most alloc//50
            lam = max(lam, 4)

            # CMA-ES state
            mean = x.copy()
            f_mean = fx
            C = np.eye(dim)
            # Inverse square root of C
            evals, vecs = np.linalg.eigh(C)
            evals = np.clip(evals, 1e-20, None)
            C_sqrt_inv = (vecs * (1.0 / np.sqrt(evals))) @ vecs.T
            step_size = sigma

            # Evolution paths
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            # Learning rates
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + mu_eff - 1)) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            iters = 0
            while remaining() >= lam and time_ok():
                # Sample
                z = rng.standard_normal((lam, dim))
                # Transform to Mahalanobis
                y = z @ C_sqrt_inv
                x_pop = mean + step_size * y
                x_pop = np.clip(x_pop, -5, 5)

                # Evaluate
                f_pop = np.empty(lam)
                for i in range(lam):
                    f_pop[i] = func(x_pop[i])
                    call_count += 1
                    if f_pop[i] < best_f:
                        best_f = f_pop[i]
                        best_x = x_pop[i].copy()
                    if not time_ok():
                        break

                if call_count >= total_budget:
                    break

                # Sort
                order = np.argsort(f_pop)
                y_sorted = y[order]
                f_sorted = f_pop[order]

                # Update mean
                new_mean = mean + step_size * (weights @ y_sorted)
                mean = new_mean

                # Update sigma
                norm_y = np.sqrt(np.sum((y_sorted[:mu] * C_sqrt_inv) ** 2, axis=1))
                h_sigma = (np.sqrt((1 - np.exp(-2 / (iters + 1) * c_sigma))) *
                           (1 - np.exp(-(iters + 1) / d_sigma * c_sigma)) *
                           (1 + np.sqrt(1 - np.exp(-2 / (iters + 1) * c_sigma))))
                p_sigma = (1 - c_sigma) * p_sigma + h_sigma * c_sigma * chi_n * (weights @ y_sorted[:mu])
                sigma_ratio = np.linalg.norm(p_sigma) / chi_n
                step_size *= np.exp(c_sigma / d_sigma * (sigma_ratio - 1))
                step_size = np.clip(step_size, 1e-12, 10.0)

                # Update C
                delta = (np.linalg.norm(mean) - np.linalg.norm(mean - step_size * (weights @ y_sorted)))
                h_c = (np.sqrt((1 - np.exp(-2 / (iters + 1) * c_c))) *
                       (1 - np.exp(-(iters + 1) / c_c * 2)))
                p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c)) * chi_n * (weights @ y_sorted[:mu])
                C = ((1 + c_1 * (2 - c_1 - c_mu) - c_mu) * C +
                     c_1 * np.outer(p_c, p_c) +
                     c_mu * step_size ** 2 * (weights[:, None] * y_sorted[:mu]) @ (weights[:, None] * y_sorted[:mu]).T)

                # Update C_sqrt_inv
                evals, vecs = np.linalg.eigh(C)
                evals = np.clip(evals, 1e-20, None)
                C_sqrt_inv = (vecs * (1.0 / np.sqrt(evals))) @ vecs.T

                iters += 1

            # Final local polish around best_x if we have budget
            if time_ok() and remaining() > 10:
                polish_budget = min(remaining(), 50)
                # Simple local refinement with decreasing step
                x = best_x.copy()
                fx = best_f
                for _ in range(polish_budget):
                    if not time_ok() or remaining() <= 0:
                        break
                    step = 0.1 * (0.9 ** _)
                    y = np.clip(x + step * rng.standard_normal(dim), -5, 5)
                    fy = func(y)
                    call_count += 1
                    if fy < fx:
                        x, fx = y, fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    else:
                        # Try a smaller step in a random direction
                        y2 = np.clip(x + step * 0.1 * rng.standard_normal(dim), -5, 5)
                        fy2 = func(y2)
                        call_count += 1
                        if fy2 < fx:
                            x, fx = y2, fy2
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()

        return best_x, best_f
