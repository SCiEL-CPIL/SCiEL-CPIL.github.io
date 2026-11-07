"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-start CMA-ES style optimizer with adaptive population and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        self.dim = self.dim
        budget = self.budget
        dim = self.dim

        # Track global best
        global_best_x = None
        global_best_f = np.inf

        # Multi-start strategy: split budget into several restarts
        # Each restart uses a CMA-ES-like population-based search
        num_restarts = 5
        budget_per_restart = max(10, budget // num_restarts)

        for restart in range(num_restarts):
            # Initial point: for first restart use center, others random
            if restart == 0:
                x0 = np.zeros(dim)
            else:
                x0 = rng.uniform(-4, 4, dim)

            # Evaluate initial point
            fx0 = func(x0)
            if fx0 < global_best_f:
                global_best_f = fx0
                global_best_x = x0.copy()

            # CMA-ES parameters
            pop_size = max(4, int(4 + 3 * np.log(dim)))
            sigma = 1.5  # Initial step size
            c1 = (2 / ((dim + 1)**2 + 10))  # Learning rate for covariance
            cmu = 2 * ((1 - c1) - 2 / ((dim + 1)**2 - 2)) / ((dim + 1)**2 + 12)
            cmu = max(cmu, 0)
            cs = (mu + 1) / (dim + mu + 5)  # Path length adaptation
            c_sigma = (1 + 2 * max(0, (mu - 1) / (dim + 1)) + cs) / (dim + 3)
            d_sigma = 1 + 2 * max(0, (mu - 1) / (dim + 1)) + cs
            mu = pop_size // 2  # Number of parents

            # Covariance matrix
            C = np.eye(dim)
            # Path accumulators
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            # Mean
            mean = x0.copy()
            best_local_f = fx0
            best_local_x = x0.copy()

            # Evaluate initial population
            for _ in range(budget_per_restart - 1):
                # Generate population
                offspring = []
                for i in range(pop_size):
                    # Sample from multivariate normal with current C and sigma
                    z = rng.multivariate_normal(np.zeros(dim), C, size=1).flatten()
                    y = np.clip(mean + sigma * z, -5, 5)
                    offspring.append(y)

                # Evaluate offspring
                fvals = [func(y) for y in offspring]
                fvals = np.array(fvals)

                # Sort by fitness
                idx = np.argsort(fvals)
                # Take top mu parents
                parents_idx = idx[:mu]
                parents = [offspring[i] for i in parents_idx]
                parent_fvals = fvals[parents_idx]

                # Update global best
                for i in parents_idx:
                    if fvals[i] < global_best_f:
                        global_best_f = fvals[i]
                        global_best_x = offspring[i].copy()
                    if fvals[i] < best_local_f:
                        best_local_f = fvals[i]
                        best_local_x = offspring[i].copy()

                # Weighted mean of parents
                weights = np.log(mu + 0.5) - np.log(1 + np.arange(mu))
                weights /= weights.sum()
                new_mean = np.sum([weights[i] * parents[i] for i in range(mu)], axis=0)

                # Update sigma (path length adaptation)
                p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mu) * (
                    C @ (new_mean - mean)
                ) / (sigma * np.sqrt(np.diag(C @ C).mean()) + 1e-10)
                sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / np.sqrt(dim) - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)

                # Update covariance matrix
                delta_mean = new_mean - mean
                p_c = (1 - c1) * p_c + np.sqrt(c1 * (2 - c1) * mu) * delta_mean / (sigma + 1e-10)

                C = (1 - c1 - cmu) * C + c1 * (np.outer(p_c, p_c) + (c1 - 2 * c1) * np.eye(dim))
                for i in range(mu):
                    y_i = (parents[i] - mean) / (sigma + 1e-10)
                    C += cmu * weights[i] * np.outer(y_i, y_i)

                mean = new_mean

        return global_best_x, global_best_f
# EVOLVE-BLOCK-END
