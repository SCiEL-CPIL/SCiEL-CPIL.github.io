"""Population-based CMA-ES for BBOB f15 (Rastrigin rotated) with adaptive restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES with adaptive restarts for Rastrigin f15."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        call_count = 0

        def eval_point(x):
            nonlocal best_x, best_f, call_count
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # ---- Phase 1: LHS exploration to locate a good basin ----
        explore_budget = max(10, min(30, budget // 12))
        n_lhs = min(explore_budget, 15)
        for i in range(n_lhs):
            x = np.zeros(dim)
            for d in range(dim):
                lo = -5.0 + d * 10.0 / dim
                hi = lo + 10.0 / dim
                x[d] = lo + rng.random() * (hi - lo)
            eval_point(x)
        for _ in range(explore_budget - n_lhs):
            x = rng.uniform(-5.0, 5.0, dim)
            eval_point(x)

        if best_x is None:
            eval_point(np.zeros(dim))

        # ---- Phase 2: Population-based CMA-ES with adaptive restarts ----
        # Use a population-based approach to maintain diversity and escape local minima
        remaining = budget - call_count
        if remaining <= 0:
            return best_x, best_f

        # CMA-ES parameters
        lambda_pop = 10  # population size
        mu = lambda_pop // 2  # number of parents
        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Initial sigma based on distance from best point
        sigma = 1.0
        x_mean = best_x.copy()

        # Covariance matrix (initially identity)
        C = np.eye(dim)
        # Inverse of C for updates
        C_inv = np.eye(dim)
        # Square root of C
        C_sqrt = np.eye(dim)
        C_sqrt_inv = np.eye(dim)

        # Step size control parameters
        c_sigma = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 3 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 2) / (dim + 2) - 1))

        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Track best for restart detection
        best_in_run = best_f
        no_improve_steps = 0
        max_no_improve = 40

        # Main CMA-ES loop
        while call_count < budget and time.time() < deadline:
            # Check for restart
            if no_improve_steps >= max_no_improve:
                # Perturb the mean to escape local minimum
                x_mean = np.clip(x_mean + rng.normal(0, 2.0, dim), -5.0, 5.0)
                sigma = 1.5
                C = np.eye(dim)
                C_inv = np.eye(dim)
                C_sqrt = np.eye(dim)
                C_sqrt_inv = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                best_in_run = best_f
                no_improve_steps = 0
                continue

            # Generate population
            population = []
            fitnesses = []
            for i in range(lambda_pop):
                # Sample from multivariate normal
                z = rng.standard_normal(dim)
                # Transform using C_sqrt
                y = x_mean + sigma * C_sqrt @ z
                y = np.clip(y, -5.0, 5.0)
                population.append(y)
                fy = eval_point(y)
                fitnesses.append(fy)

            if call_count >= budget or time.time() > deadline:
                break

            # Sort by fitness
            indices = np.argsort(fitnesses)
            population = [population[i] for i in indices]
            fitnesses = [fitnesses[i] for i in indices]

            # Update best
            if fitnesses[0] < best_f:
                best_f = fitnesses[0]
                best_x = population[0].copy()
                no_improve_steps = 0
            else:
                no_improve_steps += 1

            # Update mean
            x_mean_old = x_mean.copy()
            x_mean = np.sum(weights[:, None] * np.array(population[:mu]), axis=0)

            # Update evolution paths
            # p_sigma
            p_sigma = (1 - c_sigma) * p_sigma
            if sigma > 1e-10:
                p_sigma += np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv @ (x_mean - x_mean_old) / sigma

            # p_c
            p_c = (1 - c_c) * p_c
            if sigma > 1e-10:
                norm_p_sigma = np.linalg.norm(p_sigma)
                h_sigma = norm_p_sigma / np.sqrt(1 - (1 - c_sigma) ** (2 * (call_count - explore_budget))) if call_count > explore_budget else 1.0
                h_sigma = min(h_sigma, 1.4 + 2 / (dim + 1))
                p_c += c_c * (1 - c_c) * h_sigma * C_sqrt_inv @ (x_mean - x_mean_old) / sigma

            # Update covariance matrix
            v = C_sqrt_inv @ (x_mean - x_mean_old) / sigma if sigma > 1e-10 else np.zeros(dim)
            C = (1 + c_c * (2 - c_c) * h_sigma * (1 - c_c) / c_c) * C  # This is approximate; proper CMA-ES uses more terms
            # For simplicity and stability, we'll use a simplified update
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(p_c, p_c) + (1 - (1 - c_c) ** 2) * C)
            C += c_mu * np.sum(weights[:mu, None, None] * np.outer(population[:mu] - x_mean, population[:mu] - x_mean) / (sigma ** 2), axis=0)

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            # Add small regularization for numerical stability
            C += 1e-8 * np.eye(dim)

            # Update sigma
            if norm_p_sigma > 0 and sigma > 1e-10:
                sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 0.05, 5.0)

            # Update C_sqrt via eigenvalue decomposition
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)  # Ensure positive
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
            except:
                # Fallback to identity if eigenvalue decomposition fails
                C_sqrt = np.eye(dim)
                C_sqrt_inv = np.eye(dim)
                C_inv = np.eye(dim)
                C = np.eye(dim)

        # ---- Phase 3: Fine polish around the best point with small steps ----
        if call_count < budget and best_x is not None and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            successes = 0
            total_steps = 0
            window_size = 8
            for _ in range(budget - call_count):
                if call_count >= budget:
                    break
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(dim), -5.0, 5.0)
                fy = eval_point(y)
                total_steps += 1
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                if total_steps % window_size == 0:
                    success_rate = successes / window_size
                    if success_rate > 0.5:
                        sigma *= 1.1
                    elif success_rate < 0.2:
                        sigma *= 0.8
                    successes = 0
                sigma = np.clip(sigma, 1e-4, 0.5)

        return best_x, best_f
# EVOLVE-BLOCK-END
