"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    CMA-ES style optimizer for the Bent Cigar function (BBOB f12).
    Uses a population-based search with adaptive step size and rotation
    to handle the ill-conditioned landscape.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        dim = self.dim
        rng = np.random.default_rng(42)

        calls_used = 0
        best_x = np.zeros(dim)
        best_fx = float('inf')

        def make_call(x):
            nonlocal calls_used, best_x, best_fx
            if time.time() >= deadline:
                return None
            fx = func(x)
            calls_used += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        # CMA-ES parameters
        # Population size
        lambda_ = 4 + int(3 * np.log(dim))  # ~14 for dim=5
        mu = lambda_ // 2  # Number of best points to use for update
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights**2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))

        # Initial step size
        sigma = 1.0

        # Initial mean
        m = rng.uniform(-5, 5, dim)

        # Covariance matrix (initially identity)
        C = np.eye(dim)

        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Check function
        def check_time():
            return time.time() < deadline

        # Phase 1: CMA-ES
        while calls_used < self.budget and check_time():
            # Generate population
            population = []
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                # Apply Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                x_i = m + sigma * (L @ z)
                x_i = np.clip(x_i, -5, 5)
                population.append(x_i)

            # Evaluate population
            fitness = []
            for x_i in population:
                fx = make_call(x_i)
                if fx is None:
                    break
                fitness.append(fx)

            if len(fitness) < lambda_:
                break

            # Sort by fitness
            indices = np.argsort(fitness)
            population = [population[i] for i in indices]
            fitness = [fitness[i] for i in indices]

            # Update mean
            m_new = sum(weights[i] * population[i] for i in range(mu))
            m_new = np.clip(m_new, -5, 5)

            # Update evolution paths
            # First, compute the inverse square root of C for the step
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            except np.linalg.LinAlgError:
                C_sqrt_inv = np.eye(dim)

            # Normalized step
            step = C_sqrt_inv @ (m_new - m)
            step_norm = np.linalg.norm(step)

            # Update p_sigma
            h_sigma = (
                np.linalg.norm(step) / np.sqrt(1.0 - (1.0 - c_sigma) ** (2 * (calls_used // lambda_)))
            )
            h_sigma = min(1.0, h_sigma / (1 + 1.5 / mu_eff))
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * step

            # Update p_c
            h_c = (
                np.linalg.norm(p_sigma) / np.sqrt(1.0 - (1.0 - c_sigma) ** (2 * (calls_used // lambda_)))
            )
            h_c = min(1.0, h_c / (1 + 1.5 / mu_eff))
            p_c = (1 - c_c) * p_c + h_c * np.sqrt(c_c * (2 - c_c) * mu_eff) * step

            # Update covariance matrix
            delta_hc = (1 - h_c) * c_c * (2 - c_c)
            rank_one = np.outer(step, step)
            rank_mu = sum(
                weights[i] * np.outer(C_sqrt_inv @ (population[i] - m), C_sqrt_inv @ (population[i] - m))
                for i in range(mu)
            )
            C = (
                (1 + c_1 * delta_hc - c_1 - c_mu) * C
                + c_1 * (np.outer(p_c, p_c) + (1 - h_c) * c_c * (2 - c_c) * C)
                + c_mu * rank_mu
            )

            # Symmetrize
            C = (C + C.T) / 2

            # Update sigma
            sigma = sigma * np.exp(
                c_sigma / mu_eff * (np.linalg.norm(p_sigma) / np.sqrt(2 * dim) - 1)
            )
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Update mean
            m = m_new

            # Convergence check
            if sigma < 1e-6 or np.all(np.abs(C - np.eye(dim)) < 1e-8):
                break

        # Phase 2: Fine-tuning with adaptive 1+1 ES
        if check_time() and calls_used < self.budget:
            x = best_x.copy()
            fx = best_fx
            sigma = 0.1
            successes = 0
            total = 0

            while calls_used < self.budget and check_time():
                direction = rng.standard_normal(dim)
                y = np.clip(x + sigma * direction, -5, 5)
                fy = make_call(y)
                if fy is None:
                    break

                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1
                total += 1

                if total >= 5:
                    rate = successes / total
                    if rate > 0.3:
                        sigma *= 1.2
                    elif rate < 0.3:
                        sigma *= 0.7
                    successes = 0
                    total = 0

                sigma = np.clip(sigma, 1e-8, 1.0)
                if sigma < 1e-5:
                    break

        return best_x, best_fx

# EVOLVE-BLOCK-END
