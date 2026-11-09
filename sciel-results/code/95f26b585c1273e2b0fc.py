"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with adaptive step size and restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # We'll track the best point and its value globally
        best_x = None
        best_f = np.inf

        # Total evaluations used
        evals_used = 0

        # CMA-ES parameters
        lambda_ = 4 * (dim + 2)  # population size
        c1 = (mu + 2) / ((dim + 2) ** 2 + mu)
        cmu = 2 * (mu - 1 + 2 / (mu + 1)) / ((dim + 2) ** 2 + mu)
        c_sigma = (mu + 0.5) / (dim + 4)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        mu = lambda_ // 2  # number of selected parents

        # We'll use multiple restarts with different initializations
        # Each restart gets a fraction of the budget
        num_restarts = 3
        budget_per_restart = budget // num_restarts

        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            if evals_used >= budget:
                break

            # Initialize
            if restart == 0:
                # First restart: start near origin (likely good for shifted functions)
                x = np.zeros(dim)
            else:
                # Subsequent restarts: random points
                x = np.random.uniform(-4, 4, dim)

            fx = func(x)
            evals_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

            sigma = 1.0
            # Covariance matrix
            C = np.eye(dim)
            # Evolution paths
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            # Step sizes for paths
            # We'll use a simplified version without full CMA-ES to save time
            # Actually, let's use a simpler approach: (1+lambda)-ES with adaptation

            # Use a population-based approach
            for gen in range(budget_per_restart):
                if time.time() > deadline:
                    break
                if evals_used >= budget:
                    break

                # Generate lambda offspring
                offspring = []
                for i in range(lambda_):
                    # Simple random perturbation
                    step = sigma * np.random.standard_normal(dim)
                    y = np.clip(x + step, -5, 5)
                    fy = func(y)
                    evals_used += 1
                    offspring.append((fy, y))
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()

                if evals_used >= budget:
                    break

                # Sort by fitness
                offspring.sort(key=lambda t: t[0])

                # Update mean using weighted combination of best mu
                weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
                weights /= weights.sum()

                new_x = np.zeros(dim)
                for i in range(mu):
                    new_x += weights[i] * offspring[i][1]
                x = np.clip(new_x, -5, 5)

                # Adapt sigma based on success rate
                # Count how many of the mu best are better than previous best
                # Simple adaptation: if improvement, increase sigma; else decrease
                if offspring[0][0] < fx:
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                sigma = np.clip(sigma, 0.01, 10.0)

                fx = offspring[0][0]

        return best_x, best_f
# EVOLVE-BLOCK-END
