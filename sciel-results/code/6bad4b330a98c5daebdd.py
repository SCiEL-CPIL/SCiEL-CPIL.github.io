"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with proper implementation and time management."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _run_cma_es(self, func, x0, sigma, budget, rng, deadline):
        """Run CMA-ES from initial point x0 with step size sigma."""
        dim = self.dim
        calls_made = 0
        best_x = x0.copy()
        best_f = func(x0)
        calls_made = 1

        # CMA-ES parameters
        pop_size = 2 * (4 + int(np.log(dim)))
        mu = pop_size // 2

        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()

        # Strategy parameters
        alpha_mu = 0.54 + 0.38 * (dim / (dim + 2))
        c_sigma = (dim + 1.7) / (dim + 9.5 + alpha_mu)
        c_c = (4 + dim / dim) / (dim + 4.1 + alpha_mu)
        c_1 = 2 / ((dim + 1.3) ** 2 + alpha_mu)
        c_mu = min(1 - c_1, 2 * (alpha_mu - 0.9) / ((dim + 2) ** 2 + alpha_mu))
        d_sigma = (dim + 2) + 2 * (dim / (dim + 2) - 1)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        x = x0.copy()
        step_size = sigma
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        while calls_made < budget:
            if time.time() > deadline:
                break

            # Generate population
            offspring = []
            offspring_f = []

            for i in range(pop_size):
                if time.time() > deadline:
                    break
                z = rng.standard_normal(dim)
                y = step_size * (B @ (D * z))
                y = np.clip(x + y, -5, 5)
                fy = func(y)
                calls_made += 1
                offspring.append(y)
                offspring_f.append(fy)

            if len(offspring) == 0:
                break

            # Sort offspring
            idx = np.argsort(offspring_f)
            offspring = [offspring[i] for i in idx]
            offspring_f = [offspring_f[i] for i in idx]

            # Update best
            if offspring_f[0] < best_f:
                best_f = offspring_f[0]
                best_x = offspring[0].copy()

            # Update mean
            new_x = sum(w * offspring[i] for i, w in enumerate(weights))

            # Update sigma
            y_mean = (new_x - x) / step_size
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
            sigma_factor = np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - chi_n))
            step_size = np.clip(step_size * sigma_factor, 1e-10, 10)

            # Update covariance
            y_w = (new_x - x) / step_size
            rank_one = np.outer(y_w, y_w)
            rank_mu = sum((1 / weights[i]) * np.outer(y_w_i, y_w_i)
                          for i, y_w_i in enumerate([(offspring[j] - x) / step_size for j in range(mu)]))
            C = (1 - c_1 - c_mu) * C + c_1 * (rank_one + (1 - (1 - c_1 - c_mu) * (1 - (1 - c_1) / (c_1 * dim))) * np.eye(dim))
            C += c_mu * rank_mu

            # Update B and D
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-20, None)
            B = eigvecs
            D = np.sqrt(eigvals)

            # Update p_c
            y_w_mean = (new_x - x) / step_size
            p_c = (1 - c_c) * p_c + c_c * y_w_mean

            x = new_x

            # Check convergence
            if step_size < 1e-15:
                break

        return best_x, best_f, calls_made

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # Try multiple restarts to handle multimodality
        best_x = None
        best_f = np.inf
        total_calls = 0

        # First, do a quick random search to find a good starting point
        num_init = min(10, budget)
        for i in range(num_init):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            total_calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            if total_calls >= budget:
                break

        # Use remaining budget for CMA-ES from the best initial point
        remaining_budget = budget - total_calls
        if remaining_budget > 0:
            x0 = best_x.copy()
            sigma = 1.5  # Initial step size
            best_x_cma, best_f_cma, cma_calls = self._run_cma_es(
                func, x0, sigma, remaining_budget, rng, deadline
            )
            total_calls += cma_calls
            if best_f_cma < best_f:
                best_f = best_f_cma
                best_x = best_x_cma

        return best_x, best_f
# EVOLVE-BLOCK-END
