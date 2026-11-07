"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES for continuous black-box functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size: standard is 4 + floor(3 * log(dim))
        # For dim=5, log(5)~1.6, 3*1.6=4.8, floor=4. 4+4=8.
        # Let's use 10 to balance exploration and generation count (500/10 = 50 gens).
        pop_size = 10
        
        # Initial mean and sigma
        # Start at center of the box
        mean = np.zeros(dim)
        sigma = 2.0 # Initial step size, box is [-5, 5] so width 10
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Learning rates
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        calls = 0
        best_val = np.inf
        best_pos = mean.copy()
        gen = 0
        
        while calls < budget:
            # Generate population
            offspring = np.zeros((pop_size, dim))
            fitness = np.zeros(pop_size)
            
            # Update sampling matrices (Simplified: recompute every step for dim=5, cheap enough)
            # Actually, for dim=5, we can just use Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Regularize if not positive definite
                C += 1e-6 * np.eye(dim)
                L = np.linalg.cholesky(C)
            
            for i in range(pop_size):
                if calls >= budget:
                    break
                
                z = np.random.randn(dim)
                x = mean + sigma * (L @ z)
                
                # Clip to bounds
                x = np.clip(x, -5, 5)
                
                f_val = func(x)
                calls += 1
                fitness[i] = f_val
                offspring[i] = x
                
                if f_val < best_val:
                    best_val = f_val
                    best_pos = x.copy()
            
            if calls >= budget:
                break
                
            # Sort offspring by fitness
            sorted_indices = np.argsort(fitness)
            y_sorted = offspring[sorted_indices]
            f_sorted = fitness[sorted_indices]
            
            # Update mean
            # Weighted sum of the best mu individuals
            mean_new = np.sum(weights[:, np.newaxis] * y_sorted[:mu], axis=0)
            
            # Calculate y_tilde (mean of selected points in standard normal space)
            # y_tilde = (mean_new - mean) / sigma
            y_tilde = (mean_new - mean) / sigma
            
            # Update sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * y_tilde
            
            # h_sigma condition
            norm_p_sigma = np.linalg.norm(p_sigma)
            # gen is incremented at the end, so current gen is 'gen'
            # The exponent in the condition is related to the number of updates
            # Standard formula: 1 - (1-c_sigma)^(2 * (gen+1)) approx 2 * gen * c_sigma for small c
            # Let's use the exact form or approximation
            exp_term = 1 - (1 - c_sigma)**(2 * (gen + 1))
            if exp_term > 0:
                h_sigma = 1 if (norm_p_sigma / np.sqrt(exp_term) < (1.4 + 2 / (dim + 1)) * chi_n) else 0
            else:
                h_sigma = 1
            
            # Update C
            # p_c update
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_tilde
            
            # Rank-one and rank-mu updates
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            
            # Rank-one update
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c)
            
            # Rank-mu update
            # C = ... + c_mu * sum(wi * (zi * zi^T))
            # Where zi = (xi - mean_old) / sigma
            for i in range(mu):
                zi = (y_sorted[i] - mean) / sigma
                C += c_mu * weights[i] * np.outer(zi, zi)
            
            # Update mean
            mean = mean_new
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            
            gen += 1
            
        return best_pos, best_val
# EVOLVE-BLOCK-END
