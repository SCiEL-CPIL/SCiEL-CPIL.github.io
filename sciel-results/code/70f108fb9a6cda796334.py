"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive step size for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(42)

        # Global best
        best_x = None
        best_f = np.inf

        # Phase 1: Latin hypercube / grid sampling for global exploration
        # Use a stratified sampling approach
        n_phase1 = 50
        # Create stratified samples
        samples = np.zeros((n_phase1, dim))
        for i in range(n_phase1):
            for d in range(dim):
                # Stratified: divide each dimension into n_phase1 bins
                bin_idx = (i + d) % n_phase1
                lo = -5.0 + (10.0 / n_phase1) * bin_idx
                hi = -5.0 + (10.0 / n_phase1) * (bin_idx + 1)
                samples[i, d] = rng.uniform(lo, hi)

        for i in range(n_phase1):
            x = samples[i]
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Phase 2: CMA-ES from the best point found
        x = best_x.copy()
        fx = best_f

        # Adaptive sigma
        sigma = 1.5
        c_sigma = 1.0 / (dim + 4)
        c_c = (4 + dim / dim) / (4 + dim + dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + dim)
        c_mu = min(1 - c_1, 2 * ((dim + 1) / (dim + 3)) ** 2 - 1)

        lambda_ = 4 + int(3 * np.log(dim))  # population size
        mu = lambda_ // 2  # number of parents

        # Covariance matrix
        C = np.eye(dim)
        # Eigen decomposition for step length
        D = np.ones(dim)
        B = np.eye(dim)

        # Step size control
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        calls_used = n_phase1
        remaining = budget - n_phase1

        # CMA-ES main loop
        gen = 0
        while calls_used < budget:
            # Check deadline
            if time.time() > deadline - 5.0:
                break

            # Generate population
            population = []
            for i in range(lambda_):
                z = rng.standard_normal(dim)
                y = x + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                population.append(y)

            # Evaluate
            fvals = np.array([func(y) for y in population])
            calls_used += lambda_

            # Sort by fitness
            idx = np.argsort(fvals)
            population = [population[i] for i in idx]
            fvals = fvals[idx]

            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = population[0].copy()

            # Weighted mean of best mu
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            x_old = x.copy()
            x = sum(weights[i] * population[i] for i in range(mu))

            # Update evolution paths
            delta_x = (x - x_old) / (sigma * np.sqrt(np.diag(C)))
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * delta_x
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (gen + 1))) / chi_n < 1.4 + 2 / (dim + 1)
            h_sigma = float(h_sigma)

            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * delta_x

            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_mu = np.outer(p_c, p_c) + delta_h * C
            C = (1 + c_1 * (mu_eff - 2) - c_1) * C + c_1 * sum(
                weights[i] * np.outer(population[i] - x_old, population[i] - x_old) / (sigma ** 2)
                for i in range(mu)
            ) + c_1 * rank_mu

            # Ensure symmetry
            C = (C + C.T) / 2

            # Eigen decomposition (recompute periodically for stability)
            if gen % 5 == 0:
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    # Clip negative eigenvalues
                    eigvals = np.clip(eigvals, 1e-20, None)
                    D = 1.0 / np.sqrt(eigvals)
                    B = eigvecs
                except np.linalg.LinAlgError:
                    D = np.ones(dim)
                    B = np.eye(dim)

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100.0)

            gen += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
