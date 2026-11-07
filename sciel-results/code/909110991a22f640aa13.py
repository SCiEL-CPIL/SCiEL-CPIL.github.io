"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for Rastrigin rotated."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 285.0  # Safety margin

        def time_ok():
            return time.time() < deadline

        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * log(n))
        lam = max(4, 4 + int(3 * np.log(dim)))
        # Weighted recombination weights
        mu = lam // 2  # number of individuals contributing to update
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Step size control parameters
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        # Initial parameters
        x_mean = np.zeros(dim)
        sigma = 2.0  # Initial step size for [-5,5]^5
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        calls = 0
        best_x = x_mean.copy()
        best_fx = np.inf
        archive = []  # Store all evaluated points

        while calls < budget and time_ok():
            # Generate population
            population = []
            for i in range(lam):
                if not time_ok() or calls >= budget:
                    break
                # Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    L = np.eye(dim)
                
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                population.append((y, fy, z))
                archive.append((y, fy))
            
            if not population:
                break
            
            # Sort by fitness
            population.sort(key=lambda t: t[1])
            
            # Update best
            if population[0][1] < best_fx:
                best_fx = population[0][1]
                best_x = population[0][0].copy()
            
            # Weighted recombination
            y_w = np.zeros(dim)
            for i in range(min(mu, len(population))):
                y_w += weights[i] * population[i][0]
            
            # Update mean
            x_mean = y_w
            
            # Update step size control (sigma)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (L @ (y_w - x_mean))
            
            # Check for sigma update
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls + 1))) < (1.4 + 2 / (dim + 1)) * chi_n:
                sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance matrix
            pc = (1 - c_c) * pc + c_c * (y_w - x_mean) / sigma
            
            # Compute delta_heavy for h1
            delta_heavy = 1 if (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (calls + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            
            # Rank-one and rank-mu updates
            C = (1 + c_1 * delta_heavy - c_1 - c_mu) * C \
                + c_1 * (np.outer(pc, pc) + (1 - delta_heavy) * c_c * (2 - c_c) * C) \
                + c_mu / sigma ** 2 * np.sum([
                    weights[i] * np.outer(population[i][0] - x_mean, population[i][0] - x_mean)
                    for i in range(min(mu, len(population)))
                ])
            
            # Ensure C is positive semi-definite
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            if np.min(eigenvalues) < 1e-10:
                C = eigenvectors @ np.diag(np.maximum(eigenvalues, 1e-10)) @ eigenvectors.T
            
            # Convergence check
            if sigma < 1e-10 or np.linalg.norm(x_mean) > 1e6:
                break
        
        # Final refinement: try to improve best_x with small perturbations
        remaining = budget - calls
        if remaining > 5 and time_ok():
            for _ in range(min(remaining - 2, 20)):
                if not time_ok() or calls >= budget - 1:
                    break
                y = best_x + 0.05 * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                if fy < best_fx:
                    best_x = y
                    best_fx = fy
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
