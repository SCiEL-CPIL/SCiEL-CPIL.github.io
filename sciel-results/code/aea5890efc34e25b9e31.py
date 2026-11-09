"""Rewritten program: CMA-ES style continuous optimizer for BBOB f20 (Schwefel)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer.

    Uses a population-based approach with adaptive covariance matrix to
    efficiently navigate the rugged landscape of the Schwefel function.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size
        lam = max(4, int(4 + 3 * np.log(dim)))
        
        # Initial point: center of the box
        x_mean = np.zeros(dim)
        
        # Initial step size
        sigma = 2.0
        
        # Covariance matrix (identity initially)
        C = np.eye(dim)
        
        # Weights and recombination weights
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Track best solution
        best_x = x_mean.copy()
        best_f = func(best_x)
        
        calls_made = 1  # We already made one function call
        
        # Precompute some constants
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))
        
        while calls_made < budget:
            # Compute Cholesky
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)
            
            # Generate and evaluate offspring
            offspring = []
            offspring_f = []
            for i in range(lam):
                if calls_made >= budget:
                    break
                z = rng.standard_normal(dim)
                x_i = x_mean + sigma * (L @ z)
                x_i = np.clip(x_i, -5, 5)
                f_i = func(x_i)
                calls_made += 1
                offspring.append(x_i)
                offspring_f.append(f_i)
                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()
            
            if len(offspring) == 0:
                break
            
            # Sort by fitness
            indices = np.argsort(offspring_f)
            offspring = [offspring[i] for i in indices]
            offspring_f = [offspring_f[i] for i in indices]
            
            # Recombination
            x_new = sum(weights[i] * offspring[i] for i in range(min(mu, len(offspring))))
            
            # Update evolution paths
            # Need inverse of L for p_sigma update
            try:
                C_inv_sqrt = np.linalg.inv(L)
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(dim)
            
            # Update sigma evolution path
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ (x_new - x_mean) / sigma
            
            # Hessian correction for sigma adaptation
            norm_p_sigma = np.linalg.norm(p_sigma)
            # Simplified h_sigma check
            h_sigma = 1.0 if norm_p_sigma / chi_n < (1 + 2 / (dim + 1)) else 0.0
            
            # Adapt sigma
            sigma *= np.exp(c_sigma * (norm_p_sigma / chi_n - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)
            
            # Update covariance matrix
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_new - x_mean) / sigma
            
            # Rank-one and rank-mu updates
            rank_one = (1 + c_1 * (h_sigma * (2 - c_c) - 1) - c_mu) * C \
                + c_1 * (np.outer(p_c, p_c) - (1 - h_sigma) * C)
            
            rank_mu = c_mu * sum(weights[i] * np.outer(offspring[i] - x_mean, offspring[i] - x_mean) for i in range(min(mu, len(offspring)))) / sigma ** 2
            
            C = rank_one + rank_mu
            
            # Ensure symmetry and positive semi-definite
            C = (C + C.T) / 2
            # Add small regularization to ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update mean
            x_mean = x_new
            x_mean = np.clip(x_mean, -5, 5)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
