"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for BBOB-style Sphere f1."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        deadline = t0 + 280.0

        rng = np.random.default_rng(42)

        # CMA-ES parameters
        lam = 4 + int(3 * np.log(dim))  # population size
        mu = lam // 2  # selection size

        # Weights for weighted recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()

        # Variance reduction factor
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Step size adaptation
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

        # Covariance matrix adaptation
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

        # Initial values
        x = rng.uniform(-5, 5, dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        best_x = x.copy()
        best_f = func(x)
        calls_used = 1

        while calls_used < budget:
            # Check time
            if time.time() > deadline - 10:
                break

            # Generate population
            offspring = []
            for i in range(lam):
                if time.time() > deadline - 10:
                    break
                if calls_used >= budget:
                    break
                z = rng.standard_normal(dim)
                y = np.clip(x + sigma * z, -5, 5)
                fy = func(y)
                calls_used += 1
                offspring.append((fy, y))
            
            if len(offspring) == 0:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[0])

            # Update mean with weighted recombination
            x_new = sum(weights[i] * offspring[i][1] for i in range(min(mu, len(offspring))))

            # Update step size
            y_mean = (x_new - x) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (calls_used + 1))
                    < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1)))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(dim) - 1))

            # Update covariance matrix
            pc = (1 - c_c) * pc + hsig * c_c * (x_new - x) / sigma
            C = ((1 + c_c * (2 - c_c) - c_1 - c_mu) * C
                 + c_1 * (2 - c_c) * (np.outer(pc, pc) - C)
                 + c_mu * sum(weights[i] * np.outer(offspring[i][1] - x,
                                                   offspring[i][1] - x)
                              for i in range(min(mu, len(offspring)))
                              ) / sigma ** 2)

            # Ensure C is symmetric and positive semi-definite
            C = (C + C.T) / 2
            try:
                C = np.linalg.cholesky(C)
                C = np.dot(C, C.T)
            except:
                C = np.eye(dim)

            x = x_new

            # Update best
            if offspring[0][0] < best_f:
                best_f = offspring[0][0]
                best_x = offspring[0][1].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
