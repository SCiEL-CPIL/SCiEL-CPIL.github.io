"""BBOB f14 (Different powers) optimizer: CMA-ES with time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer for BBOB f14 on [-5,5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 320.0  # safety margin before 330s kill

        rng = np.random.default_rng(12345)

        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(dim)))  # ~9 for dim=5
        lambda_ = pop_size

        # Initial population: spread out
        x = np.zeros(dim)
        sigma = 2.0

        # CMA-ES state
        C = np.eye(dim)
        C_inv_sqrt = np.eye(dim)
        Cs = np.zeros((dim, dim))
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Step size control
        cs = (lambda_ + 2) / (dim + lambda_ + 5)
        cmu = min(0.5, (lambda_ - 2 + 1.0 / (lambda_ + 2)) / (dim + lambda_ + 2) - 1)
        ds = 1 + 2 * max(0, np.sqrt((lambda_ - 1) / (dim + 1)) - 1) + cs
        hsigma = (1.4 + 2 / (dim + 1)) * np.sqrt((1 + 1 / cs) / (1 - 1 / cs))

        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        best_x = x.copy()
        best_f = np.inf
        call_count = 0

        def evaluate(point):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            val = func(point)
            call_count += 1
            return val

        # Initial evaluation
        f0 = evaluate(x)
        if np.isfinite(f0):
            best_f = f0
            best_x = x.copy()

        # Main CMA-ES loop
        while call_count < budget and time.time() < deadline:
            # Generate population
            population = []
            for i in range(lambda_):
                # Sample from N(x, sigma^2 * C)
                z = rng.standard_normal(dim)
                # Apply C^(1/2) via Cholesky-like (use eigen decomposition periodically)
                # For simplicity, use the standard CMA-ES update with C matrix
                # We'll use the fact that C is symmetric positive definite
                # Compute C^(1/2) * z using eigendecomposition (expensive but dim=5 is small)
                # Actually, let's just use a simpler approach: maintain C and use it directly
                # For dim=5, we can do a quick eigendecomposition
                vals, vecs = np.linalg.eigh(C)
                vals = np.maximum(vals, 1e-12)
                C_sqrt = vecs @ np.diag(np.sqrt(vals)) @ vecs.T
                y = x + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                population.append(y)

            # Evaluate population
            fitness = []
            for y in population:
                f = evaluate(y)
                fitness.append(f)
                if f < best_f:
                    best_f = f
                    best_x = y.copy()

            fitness = np.array(fitness)
            population = np.array(population)

            # Sort by fitness
            order = np.argsort(fitness)
            population = population[order]
            fitness = fitness[order]

            # Weighted recombination
            weights = np.ones(lambda_) / lambda_
            # Use log-weights for better performance
            weights = np.exp(-np.arange(1, lambda_ + 1) / (lambda_ / 2))
            weights /= weights.sum()

            mu_eff = 1.0 / np.sum(weights[:lambda_ // 2 + 1] ** 2)

            # Update mean
            x_old = x.copy()
            x = np.sum(weights[:, None] * population, axis=0)

            # Update sigma (CSA)
            d_norm = np.linalg.norm(x - x_old)
            if sigma > 0 and d_norm > 0:
                ps = (1 - cs) * ps + cs * np.sqrt(cs * (2 - cs) * mu_eff) * (x - x_old) / (sigma * d_norm)
            else:
                ps = np.zeros(dim)

            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (budget - call_count + 1) / lambda_)) < (hsigma + 0.3) * chi_n:
                pc = (1 - cs) * pc + cs * d_norm / sigma
                C = (1 - cs * (2 - cs) * cmu) * C + cs * (2 - cs) * cmu * np.outer(pc, pc)

            # Update covariance matrix
            diff = population - x_old
            C = (1 - cs * (2 - cs) * (cmu + 1 / mu_eff)) * C \
                + cs * (2 - cs) * cmu * np.sum(weights[:, None, None] * diff[:, :, None] * diff[:, None, :], axis=0) \
                + cs * (2 - cs) * (1 / mu_eff - cmu) * np.outer(pc, pc)

            # Ensure positive definiteness
            vals, vecs = np.linalg.eigh(C)
            vals = np.maximum(vals, 1e-12)
            C = vecs @ np.diag(vals) @ vecs.T

            # Adapt sigma
            if d_norm > 1e-12:
                sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
                sigma = np.clip(sigma, 1e-10, 50)

        return best_x, best_f
# EVOLVE-BLOCK-END
