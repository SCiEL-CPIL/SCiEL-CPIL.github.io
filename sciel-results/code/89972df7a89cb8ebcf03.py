"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for BBOB-style f6 (Attractive sector)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        # Standard lambda for d=5 is 4 + 3*ln(5) ~ 8. 
        # Let's use 10 for slightly better coverage per generation.
        lambda_pop = max(4, int(4 + 3 * np.log(d)) + 2) 
        mu = lambda_pop // 2  # number of recombination weights

        # Compute recombination weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (d + 1)) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))

        # Initial state
        x_mean = np.zeros(d)
        # Start with a larger sigma to ensure we explore the [-5, 5] box well initially
        # f6 is often scaled such that the minimum is not at 0, but the box is [-5,5].
        # A sigma of 2.0 covers most of the box.
        sigma = 2.0
        C = np.eye(d)
        C_inv_sqrt = np.eye(d)
        C_sqrt = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        def update_cholesky(C):
            """Compute square root and inverse square root of C via Cholesky."""
            # Add small regularization for stability
            C_reg = C + 1e-12 * np.eye(d)
            try:
                L = np.linalg.cholesky(C_reg)
                C_sqrt = L
                C_inv_sqrt = np.linalg.inv(L).T
            except np.linalg.LinAlgError:
                # Fallback to eigenvalue decomposition
                evals, evecs = np.linalg.eigh(C_reg)
                evals = np.maximum(evals, 1e-12)
                C_sqrt = evecs @ np.diag(np.sqrt(evals))
                C_inv_sqrt = evecs @ np.diag(1.0 / np.sqrt(evals))
            return C_sqrt, C_inv_sqrt

        update_cholesky(C)

        best_x = None
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal best_x, best_f, calls_made
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # Initial population
        for i in range(lambda_pop):
            z = rng.standard_normal(d)
            x_i = x_mean + sigma * (C_sqrt @ z)
            evaluate(x_i)

        # Main loop
        while calls_made < budget:
            # Generate population
            offspring = []
            fitnesses = []
            for i in range(lambda_pop):
                if calls_made >= budget:
                    break
                z = rng.standard_normal(d)
                x_i = x_mean + sigma * (C_sqrt @ z)
                f_i = evaluate(x_i)
                offspring.append(x_i)
                fitnesses.append(f_i)

            if len(offspring) == 0:
                break

            # Sort by fitness
            idx = np.argsort(fitnesses)
            offspring = [offspring[i] for i in idx]
            fitnesses = [fitnesses[i] for i in idx]

            # Recombination
            x_new = np.zeros(d)
            for i in range(min(mu, len(offspring))):
                x_new += weights[i] * offspring[i]
            x_mean_old = x_mean.copy()
            x_mean = x_new

            # Compute y_i vectors (standardized steps)
            # y_i = (x_i - x_mean_old) / sigma
            # We only need the selected ones for the covariance update
            y_vectors = []
            for i in range(min(mu, len(offspring))):
                y_i = (offspring[i] - x_mean_old) / sigma
                y_vectors.append(y_i)
            
            # Update evolution paths
            d_x = (x_mean - x_mean_old) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (4 - c_sigma) * mu_eff) * (C_inv_sqrt @ d_x)
            norm_ps = np.linalg.norm(ps)

            # Update h_sigma
            h_sigma = 1.0 if (norm_ps / chi_n < 1.4 + 2 / (d + 1)) else 0.0

            # Update pc
            pc = (1 - c_c) * pc + h_sigma * np.sqrt(c_c * (4 - c_c) * mu_eff) * d_x

            # Update covariance matrix
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc pc^T + (1-h_sigma) * c_c * (4-c_c) * C) + c_mu * sum(w_i * y_i y_i^T)
            
            # Rank-1 update part from pc
            pc_outer = np.outer(pc, pc)
            
            # Rank-mu update part from y_vectors
            y_outer_sum = np.zeros((d, d))
            for i in range(min(mu, len(y_vectors))):
                y_i = y_vectors[i]
                y_outer_sum += weights[i] * np.outer(y_i, y_i)

            # Combine
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (pc_outer + (1 - h_sigma) * c_c * (4 - c_c) * C)
            C += c_mu * y_outer_sum

            # Update step size
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))

            # Ensure positive definiteness and update Cholesky
            C = (C + C.T) / 2
            # Regularize if needed
            evals, evecs = np.linalg.eigh(C)
            if np.any(evals < 1e-12):
                evals = np.maximum(evals, 1e-12)
                C = evecs @ np.diag(evals) @ evecs.T
                C = (C + C.T) / 2
            
            update_cholesky(C)

            # Clamp sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-10, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
