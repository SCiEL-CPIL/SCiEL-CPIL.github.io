"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES style optimization with adaptive step size."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Initial point
        x = np.zeros(dim)
        fx = func(x)
        
        # CMA-ES parameters
        lambda_pop = max(10, 4 + int(3 * dim))
        sigma = 2.0
        
        # Covariance matrix
        C = np.eye(dim)
        C_inv = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Parameters
        c_sigma = (lambda_pop + 2) / (dim + lambda_pop + 5)
        mu_sigma = 0.5
        c_c = (4 + lambda_pop / dim) / (dim + 4 + 2 * lambda_pop / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + lambda_pop)
        c_mu = min(1 - c_1, 2 * (mu_sigma - 2 + 1 / mu_sigma) / ((dim + 2) ** 2 + lambda_pop))
        
        # Weights
        weights = np.log(lambda_pop / 2 + 0.5) - np.log(np.arange(1, lambda_pop + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights[:int(np.floor(lambda_pop / 2))] ** 2)
        
        def cholesky(A):
            try:
                L = np.linalg.cholesky(A)
                return L
            except np.linalg.LinAlgError:
                # Add jitter
                L = np.linalg.cholesky(A + 1e-6 * np.eye(dim))
                return L
        
        L = cholesky(C)
        
        calls_made = 1
        for gen in range(budget - 1):
            # Generate offspring
            offspring = []
            for i in range(lambda_pop):
                if calls_made >= budget:
                    break
                z = np.random.standard_normal(dim)
                y = np.clip(x + sigma * L @ z, -5, 5)
                fy = func(y)
                offspring.append((fy, y))
                calls_made += 1
            
            if not offspring:
                break
            
            # Sort offspring
            offspring.sort(key=lambda t: t[0])
            
            # Update mean
            w = weights[:len(offspring)]
            if len(w) < int(np.floor(lambda_pop / 2)):
                w = weights[:len(offspring)]
                mu_eff = 1 / np.sum(w ** 2)
            x_new = np.sum(w[:, np.newaxis] * np.array([y for _, y in offspring]), axis=0)
            
            # Update evolution paths
            diff = (x_new - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (gen + 1)) <
                       (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (mu_eff - 1)))
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * diff
            
            # Update covariance
            C = (1 + c_1 * (2 * h_sigma - 1) - c_1 - c_mu) * C
            rank1 = np.outer(p_c, p_c)
            rankmu = np.sum(w[:, np.newaxis, np.newaxis] * np.array([np.outer(y - x, y - x) for _, y in offspring]) / sigma ** 2, axis=0)
            C += c_1 * (rank1 - C) + c_mu * (rankmu - C)
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update Cholesky factor
            L = cholesky(C)
            
            # Update sigma
            sigma *= np.exp(c_sigma / 5 * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Update current best
            if offspring[0][0] < fx:
                x, fx = offspring[0][1], offspring[0][0]
            else:
                x = x_new
        
        return x, fx
# EVOLVE-BLOCK-END
