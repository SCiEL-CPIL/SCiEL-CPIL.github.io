"""CMA-ES based optimizer for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES with L-BFGS-B refinement for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # Evaluation counter
        eval_count = 0

        def wrapped_func(x):
            nonlocal eval_count
            if eval_count >= budget:
                return float('inf')
            eval_count += 1
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            return float(func(x))

        best_x = np.zeros(dim)
        best_f = float('inf')

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        rng = np.random.default_rng(42)

        # Phase 1: Initial diverse sampling
        n_initial = 25
        for _ in range(n_initial):
            if eval_count >= budget - 200:
                break
            if time.time() > deadline:
                break
            pt = rng.uniform(-4.5, 4.5, dim)
            f_val = wrapped_func(pt)
            update_best(pt, f_val)

        # Phase 2: CMA-ES optimization
        # CMA-ES parameters
        pop_size = 15
        sigma = 1.5  # initial step size
        max_iter = 80
        stagnation_limit = 15  # iterations without improvement before restart

        # CMA-ES state
        mean = best_x.copy()
        C = np.eye(dim)  # covariance matrix
        C_sqrt_inv = np.eye(dim)  # inverse square root of C
        p_sigma = np.zeros(dim)  # evolution path for sigma
        p_c = np.zeros(dim)  # evolution path for C

        # Strategy parameters
        mu = pop_size // 2  # number of selected individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # CMA-ES main loop
        stagnation_count = 0
        for iteration in range(max_iter):
            if eval_count >= budget - 100:
                break
            if time.time() > deadline:
                break

            # Generate population
            population = []
            for i in range(pop_size):
                z = rng.standard_normal(dim)
                # Transform using C_sqrt_inv
                x = mean + sigma * C_sqrt_inv @ z
                x = np.clip(x, -5.0, 5.0)
                population.append(x)

            # Evaluate population
            fitness = []
            for x in population:
                if eval_count >= budget - 100:
                    fitness.append(float('inf'))
                    continue
                if time.time() > deadline:
                    fitness.append(float('inf'))
                    continue
                f_val = wrapped_func(x)
                fitness.append(f_val)
                update_best(x, f_val)

            # Sort by fitness
            order = np.argsort(fitness)
            population = [population[i] for i in order]
            fitness = [fitness[i] for i in order]

            # Check for stagnation
            if fitness[0] >= best_f * (1 + 1e-10) and best_f != float('inf'):
                stagnation_count += 1
            else:
                stagnation_count = 0

            # Update mean
            mean = sum(weights[i] * population[i] for i in range(mu))

            # Update evolution paths
            z_mean = C_sqrt_inv @ ((mean - best_x) / sigma) if sigma > 1e-10 else np.zeros(dim)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_mean
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * z_mean

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Update covariance matrix
            delta_h = (1 + c_c) * np.linalg.norm(p_c) / np.sqrt(mu_eff) <= 1.4 + 2 / (dim + 1)
            if delta_h:
                rank_one = np.outer(p_c, p_c)
                rank_mu = sum(weights[i] * np.outer(population[i] - best_x, population[i] - best_x) / sigma ** 2 for i in range(mu))
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * (rank_one + (1 - delta_h) * (np.linalg.norm(p_c) ** 2 / mu_eff - 1) * C) + c_mu * rank_mu

            # Regularize C and compute C_sqrt_inv
            C = (C + C.T) / 2  # symmetrize
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            eigenvalues = np.clip(eigenvalues, 1e-12, None)
            C = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
            C_sqrt_inv = eigenvectors @ np.diag(1 / np.sqrt(eigenvalues)) @ eigenvectors.T

            # Restart if stagnated
            if stagnation_count >= stagnation_limit:
                # Restart from best point with larger sigma
                mean = best_x.copy()
                sigma = 2.0
                C = np.eye(dim)
                C_sqrt_inv = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                stagnation_count = 0
                break

        # Phase 3: L-BFGS-B refinement from best point
        if eval_count < budget - 20 and time.time() < deadline:
            try:
                result = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 50, 'maxls': 20, 'ftol': 1e-15, 'gtol': 1e-8}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        # Phase 4: Use remaining budget for local random search
        while eval_count < budget and time.time() < deadline:
            x = best_x + rng.normal(0, 0.01, dim)
            x = np.clip(x, -5.0, 5.0)
            f_val = wrapped_func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
