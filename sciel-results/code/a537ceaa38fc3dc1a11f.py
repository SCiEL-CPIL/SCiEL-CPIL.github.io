import numpy as np


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_pop = 10
        
        # Weights
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Effective population size
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Initial parameters
        x_mean = np.zeros(dim)
        sigma = 1.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        best_x = x_mean.copy()
        best_f = float('inf')
        
        # Precompute chi_n
        chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim * dim))
        
        while budget >= lambda_pop:
            # 1. Compute eigen decomposition of C
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            sqrt_eigvals = np.sqrt(eigvals)
            inv_sqrt_eigvals = 1.0 / sqrt_eigvals
            C_inv_sqrt = (eigvecs * inv_sqrt_eigvals) @ eigvecs.T
            
            # 2. Sample lambda points
            z = rng.normal(0, 1, (lambda_pop, dim))
            x_pop = x_mean + sigma * (z * sqrt_eigvals) @ eigvecs.T
            x_pop = np.clip(x_pop, -5, 5)
            
            # 3. Evaluate
            fvals = np.array([func(x) for x in x_pop])
            budget -= lambda_pop
            
            # Update best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = x_pop[min_idx].copy()
            
            # 4. Sort
            idx_sorted = np.argsort(fvals)
            x_sorted = x_pop[idx_sorted]
            
            # 5. Update mean
            x_new_mean = np.sum(weights[:, np.newaxis] * x_sorted[:mu], axis=0)
            
            # 6. Update path for sigma (ps)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (x_new_mean - x_mean) / sigma)
            
            # 7. Update path for covariance (pc)
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * ((x_new_mean - x_mean) / sigma)
            
            # 8. Update covariance
            norm_ps = np.linalg.norm(ps)
            delta_h = (norm_ps / chi_n - (1 - c_sigma)) > 1.4 + 2 / (dim + 1)
            
            rank_one = np.outer(pc, pc) + (1 - (1 - c_sigma) ** 2) * delta_h * np.eye(dim)
            
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                diff = x_sorted[i] - x_new_mean
                rank_mu += weights[i] * np.outer(diff, diff)
                
            C = (1 - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            C = (C + C.T) / 2.0
            C += 1e-10 * np.eye(dim)
            
            # 9. Update sigma
            sigma = sigma * np.exp(c_sigma / chi_n * (norm_ps / chi_n - 1.0))
            
            # 10. Update mean
            x_mean = x_new_mean
            
            # Check convergence
            if sigma < 1e-12:
                break

        return best_x, best_f
