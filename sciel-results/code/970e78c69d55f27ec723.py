"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for black-box optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s limit

        # CMA-ES parameters
        mu = max(1, dim // 2)  # number of best solutions
        # Weights for the best mu solutions
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        c1 = (2 * (mueff + 0.5)) / ((dim + 2)**2 + mueff + 5)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff)) / ((dim + 2)**2 + mueff + 5)
        cs = (mueff + 1.5) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)

        # Population size
        lambda_ = 2 * dim
        # Initial sigma
        sigma = 2.0

        # Initial population center
        rng = np.random.default_rng(0)
        mean = rng.uniform(-4, 4, dim)

        # Covariance matrix (identity initially)
        C = np.eye(dim)
        # Inverse of C (for generating samples)
        C_inv = np.eye(dim)
        # Cholesky factor of C
        C_chol = np.eye(dim)
        # Cholesky factor of C^{-1}
        C_inv_chol = np.eye(dim)

        # Evolution path
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Track best solution found
        best_x = mean.copy()
        best_f = func(best_x)

        calls_used = 1
        call_count = 0

        while calls_used < budget and time.time() < deadline:
            # Generate population
            offspring = np.zeros((lambda_, dim))
            for i in range(lambda_):
                # Sample from N(mean, sigma^2 * C)
                z = rng.standard_normal(dim)
                # Transform using Cholesky factor
                y = C_chol @ z
                offspring[i] = mean + sigma * y

            # Clip to bounds
            offspring = np.clip(offspring, -5, 5)

            # Evaluate offspring
            fvals = np.zeros(lambda_)
            for i in range(lambda_):
                fvals[i] = func(offspring[i])
                calls_used += 1
                call_count += 1

                # Update best
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = offspring[i].copy()

                # Time check inside loop for safety
                if time.time() > deadline - 5:
                    break

            # Check if we broke out early due to time
            if time.time() > deadline - 5:
                break

            # Sort by fitness
            indices = np.argsort(fvals)
            offspring = offspring[indices]
            fvals = fvals[indices]

            # Update mean
            new_mean = np.sum(weights[:, np.newaxis] * offspring[:mu], axis=0)

            # Update evolution path for sigma (ps)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (C_inv_chol @ (new_mean - mean))

            # Update covariance matrix adaptation
            # Compute rank-one update
            delta_mean = (new_mean - mean) / sigma
            # Compute rank-mu update
            y_best = (offspring[:mu] - mean) / sigma
            delta_C = np.outer(delta_mean, delta_mean)
            for i in range(mu):
                delta_C += weights[i] * np.outer(y_best[i], y_best[i])

            # Update C
            C = (1 + c1 * cc * (np.outer(pc, pc) - C) + c1 * (np.outer(delta_mean, delta_mean) - C)
                 + cmu * delta_C)

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2

            # Update pc
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (call_count + 1)))) < \
                   (1 + 2 / (dim + 1)) * np.sqrt(1 - (1 - cs)**(2 * (call_count + 1)))
            if hsig:
                pc = (1 - cc) * pc + cc * delta_mean

            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(mueff) - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)

            # Update Cholesky factors
            try:
                C_chol = np.linalg.cholesky(C)
                C_inv = np.linalg.inv(C)
                C_inv = (C_inv + C_inv.T) / 2
                C_inv_chol = np.linalg.cholesky(C_inv)
            except np.linalg.LinAlgError:
                # If Cholesky fails, reset to identity
                C = np.eye(dim)
                C_inv = np.eye(dim)
                C_chol = np.eye(dim)
                C_inv_chol = np.eye(dim)
                sigma = max(sigma, 0.1)

            mean = new_mean

        return best_x, best_f
# EVOLVE-BLOCK-END
