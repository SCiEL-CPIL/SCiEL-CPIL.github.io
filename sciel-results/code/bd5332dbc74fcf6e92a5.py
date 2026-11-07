"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive population size."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 300s limit, 20s safety margin
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        # CMA-ES Parameters
        # Initial mean: random start
        mean = rng.uniform(-5, 5, dim)
        
        # Initial sigma: large to cover the box quickly
        sigma = 4.0
        
        # Population size: adaptive based on remaining budget
        # Standard CMA-ES uses lambda = 4 + floor(3 * log(n))
        # For n=5, lambda = 4 + 3*1.6 = 8.8 -> 8
        # We will use a smaller population to conserve budget for more generations
        # Let's try lambda = 4 initially, increasing slightly or keeping fixed.
        # With budget 500, if lambda=4, we can do ~125 generations.
        # If lambda=8, ~62 generations.
        # Early progress is key for AOCC. Smaller lambda explores less per gen but adapts slower.
        # Let's stick to a standard small population for high dimensions, but n=5 is low.
        lambda_pop = 4
        
        # Weights
        mu = lambda_pop // 2  # Number of selected individuals
        weights = np.ones(mu)
        # Standard weights: mu_i = ln(mu+0.5) - ln(i)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / (dim**1.7 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # State
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        best_x = mean.copy()
        best_f = func(best_x)
        calls = 1
        
        # Main Loop
        while calls < budget:
            if time.time() > deadline - 5.0:
                break
            
            # Generate population
            # Inverse square root of C
            try:
                L = np.linalg.cholesky(C)
                L_inv = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                # Fallback if C becomes singular
                L = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
                L_inv = np.linalg.inv(L)
            
            # Sample offspring
            offspring = []
            for i in range(lambda_pop):
                z = rng.standard_normal(dim)
                y = mean + sigma * (L_inv @ z)
                y = np.clip(y, -5, 5)
                offspring.append(y)
            
            # Evaluate offspring
            fitness = []
            for y in offspring:
                fy = func(y)
                calls += 1
                fitness.append(fy)
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                
                if calls >= budget:
                    break
            
            if calls >= budget:
                break
            
            # Sort offspring by fitness
            sorted_indices = np.argsort(fitness)
            selected = [offspring[i] for i in sorted_indices[:mu]]
            
            # Update mean
            new_mean = np.sum([w * y for w, y in zip(weights, selected)], axis=0)
            
            # Update step size
            # D = sqrt(1 - exp(-2/mu_eff) * (1 - c_sigma))
            D = np.sqrt(1 - np.exp(-2/mu_eff) * (1 - c_sigma))
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (L_inv @ (new_mean - mean))
            if np.linalg.norm(ps) / chi_n < (1.4 + 2/(dim+1)) * np.sqrt((1-c_c)/(1+c_c)):
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Update covariance matrix
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * (new_mean - mean) / sigma
            
            # Rank-1 update
            y_sigma = (new_mean - mean) / sigma
            C = (1 + c_1 * c_mu) * C - c_1 * C @ C + c_1 * (1 + (1 - nu) * c_1 * c_mu) * np.outer(pc, pc)
            
            # Rank-mu update
            H = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (selected[i] - mean) / sigma
                H += weights[i] * np.outer(y_i, y_i)
            C += c_mu * H
            
            # Symmetrize and ensure positive semi-definite
            C = (C + C.T) / 2
            # Eigenvalue check for stability
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.min(eigvals) < 1e-12:
                    C = eigvecs @ np.diag(np.maximum(eigvals, 1e-12)) @ eigvecs.T
            except:
                pass
            
            mean = new_mean
            
            # Convergence check: if sigma is very small, stop
            if sigma < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
