"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES-like optimizer with aggressive exploration for Schwefel f20."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = max(4, int(4 + 3 * np.log(dim)))  # population size
        c_sigma = (mu_eff + 2) / ((dim + 2) ** 2 + mu_eff)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / ((dim + 2) ** 2 + mu_eff)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / dim) / ((dim + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Initial population: spread out for exploration
        x_mean = rng.uniform(-3, 3, dim)
        sigma = 3.0
        B = np.eye(dim)
        D = np.ones(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        best_x = x_mean.copy()
        best_f = func(x_mean)
        calls_made = 1

        # Generate initial population
        pop = []
        for i in range(lambda_):
            x_i = x_mean + sigma * (B @ (D * rng.standard_normal(dim)))
            x_i = np.clip(x_i, -5, 5)
            f_i = func(x_i)
            pop.append((f_i, x_i))
            calls_made += 1
            if calls_made > budget:
                break

        pop.sort(key=lambda t: t[0])

        # Track progress for early stopping
        stall_count = 0
        last_improvement_f = best_f
        exploration_interval = 30
        calls_since_exploration = 0

        while calls_made < budget and time.time() < deadline:
            # Check for stall - trigger exploration
            calls_since_exploration += 1
            if best_f < last_improvement_f - 1e-6:
                stall_count = 0
                last_improvement_f = best_f
            else:
                stall_count += 1

            if stall_count >= exploration_interval:
                # Aggressive exploration: random restart in a smaller region
                x_mean = best_x + rng.uniform(-2, 2, dim)
                x_mean = np.clip(x_mean, -5, 5)
                sigma = max(0.5, sigma * 1.5)
                B = np.eye(dim)
                D = np.ones(dim)
                ps = np.zeros(dim)
                pc = np.zeros(dim)
                stall_count = 0
                calls_since_exploration = 0

            # Update mean
            w = weights[:min(mu, len(pop))]
            if len(pop) >= mu:
                x_new_mean = np.zeros(dim)
                for i in range(min(mu, len(pop))):
                    x_new_mean += w[i] * pop[i][1]
            else:
                x_new_mean = pop[0][1]

            # Update evolution paths
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (B @ (D * (x_new_mean - x_mean)))
            h_sigma = 1.0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made - lambda_))) < (1.4 + 2 / (dim + 1)) * chi_n else 0.0
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * (x_new_mean - x_mean)

            # Update covariance matrix
            delta_d = h_sigma * c_c * (1 - c_c) if h_sigma else 0.0
            outer = np.outer(x_new_mean - x_mean, x_new_mean - x_mean)
            for i in range(min(mu, len(pop))):
                y_i = (pop[i][1] - x_mean) / sigma
                outer += weights[i] * (np.outer(y_i, y_i) - np.eye(dim))
            C = (1 + c_1 * (2 - c_1) + c_mu) * B @ (D ** 2) @ B.T + c_1 * (delta_d * np.outer(pc, pc) - pc @ pc) + c_mu * (B @ (D ** 2) @ B.T) * 0 + c_mu * outer
            # Ensure positive semi-definite
            C = 0.5 * (C + C.T)
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update B and D
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            B = eigvecs
            D = np.sqrt(eigvals)

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)

            x_mean = x_new_mean
            x_mean = np.clip(x_mean, -5, 5)

            # Generate new population
            new_pop = []
            for i in range(lambda_):
                if calls_made >= budget or time.time() > deadline:
                    break
                x_i = x_mean + sigma * (B @ (D * rng.standard_normal(dim)))
                x_i = np.clip(x_i, -5, 5)
                f_i = func(x_i)
                new_pop.append((f_i, x_i))
                calls_made += 1

            if not new_pop:
                break

            new_pop.sort(key=lambda t: t[0])

            # Update best
            if new_pop[0][0] < best_f:
                best_f = new_pop[0][0]
                best_x = new_pop[0][1].copy()

            pop = new_pop

        return best_x, best_f
# EVOLVE-BLOCK-END
