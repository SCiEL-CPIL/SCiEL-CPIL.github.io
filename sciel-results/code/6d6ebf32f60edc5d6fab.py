"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with deadline guard for BBOB f21 (Gallagher 101 peaks)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 330s kill
        budget = self.budget
        dim = self.dim

        def check_time():
            return time.time() < deadline

        def safe_call(x):
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            return float(func(x))

        rng = np.random.default_rng(42)

        # LHS-based initial sampling for diversity
        n_initial = min(50, budget // 10)
        # Simple Latin Hypercube Sampling
        lhs_points = []
        for i in range(n_initial):
            point = np.zeros(dim)
            for d in range(dim):
                # Stratified sampling
                idx = (i + d) % n_initial
                point[d] = -5.0 + 10.0 * (idx + rng.random()) / n_initial
            lhs_points.append(point)

        # Evaluate initial points
        best_x = None
        best_f = float('inf')
        eval_count = 0

        for p in lhs_points:
            if not check_time():
                break
            f = safe_call(p)
            eval_count += 1
            if f < best_f:
                best_f = f
                best_x = p.copy()

        # CMA-ES implementation
        def cma_es(x0, f0, sigma0, remaining_budget, rng_local):
            """Run CMA-ES starting from x0 with initial sigma."""
            if remaining_budget <= 0:
                return x0, f0, 0

            dim_local = dim
            # Population size
            lambda_pop = max(4, int(4 + 3 * np.log(dim_local)))
            mu = lambda_pop // 2  # number of best individuals

            # Weights
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            # Step size control parameters
            c_sigma = (mu_eff + 2) / (dim_local + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim_local + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim_local) / (dim_local + 4 + 2 * mu_eff / dim_local)
            c_1 = 2 / ((dim_local + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim_local + 2) ** 2 + mu_eff))

            # Initialization
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim_local)
            C_sqrt_inv = np.eye(dim_local)
            p_sigma = np.zeros(dim_local)
            p_c = np.zeros(dim_local)
            chi_n = np.sqrt(dim_local) * (1 - 1.0 / (4 * dim_local) + 1.0 / (21 * dim_local ** 2))

            best_x_local = x0.copy()
            best_f_local = f0
            evals_local = 0

            max_iter = remaining_budget // lambda_pop
            for gen in range(max_iter):
                if not check_time():
                    break
                if evals_local + lambda_pop > remaining_budget:
                    # Do only as many as we can afford
                    lambda_pop = remaining_budget - evals_local
                    if lambda_pop <= 0:
                        break
                    mu = max(1, lambda_pop // 2)
                    weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
                    weights /= weights.sum()
                    mu_eff = 1.0 / np.sum(weights ** 2)

                # Generate population
                offspring = []
                for i in range(lambda_pop):
                    if not check_time():
                        break
                    z = rng_local.standard_normal(dim_local)
                    y = x_mean + sigma * (C_sqrt_inv @ z)
                    y = np.clip(y, -5.0, 5.0)
                    offspring.append((y, safe_call(y)))
                    evals_local += 1

                if len(offspring) < mu:
                    break

                # Sort by fitness
                offspring.sort(key=lambda t: t[1])
                selected = offspring[:mu]

                # Update mean
                new_mean = np.zeros(dim_local)
                for i, (y, f) in enumerate(selected):
                    new_mean += weights[i] * y
                x_mean = new_mean

                # Update best
                if selected[0][1] < best_f_local:
                    best_f_local = selected[0][1]
                    best_x_local = selected[0][0].copy()

                # Compute covariance update
                y_w = np.zeros((mu, dim_local))
                for i, (y, f) in enumerate(selected):
                    y_w[i] = (y - x_mean) / sigma

                # Update C
                C = (1 - c_1 - c_mu) * C + c_1 * (2 - mu_eff) * C + c_mu * (y_w.T @ y_w)

                # Recompute C_sqrt_inv via Cholesky
                try:
                    L = np.linalg.cholesky(C)
                    C_sqrt_inv = np.linalg.inv(L)
                except np.linalg.LinAlgError:
                    # Regularize
                    C += 1e-12 * np.eye(dim_local)
                    L = np.linalg.cholesky(C)
                    C_sqrt_inv = np.linalg.inv(L)

                # Update p_sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_sqrt_inv @ (y_w.T @ weights))
                if np.linalg.norm(p_sigma) > (1.4 + c_sigma) * chi_n:
                    sigma /= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

                # Update p_c
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_sqrt_inv @ (y_w.T @ weights))
                h_sig = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1))) < (1.4 + c_sigma) * chi_n else 0
                C = (1 - c_1 - c_mu + h_sig * c_mu * (mu_eff - 2) * (1 - c_c)) * C + c_1 * (2 - mu_eff) * C + c_mu * (y_w.T @ y_w)
                if h_sig:
                    C += c_mu * (2 - mu_eff) * np.outer(p_c, p_c)

                # Clamp sigma
                sigma = max(sigma, 1e-10)
                sigma = min(sigma, 10.0)

            return best_x_local, best_f_local, evals_local

        # Run CMA-ES from best initial point
        if best_x is not None and check_time() and eval_count < budget:
            remaining = budget - eval_count
            best_x, best_f, used = cma_es(best_x, best_f, 1.0, remaining, rng)
            eval_count += used

        return best_x, best_f
# EVOLVE-BLOCK-END
