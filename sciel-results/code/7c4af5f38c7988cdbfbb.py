"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = time.time() + 280

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)
        
        # CMA-ES Parameters
        lambda_pop = 12
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights ** 2).sum()
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / (dim ** 1.6 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 2 / mu_eff) / (dim + 2) ** 2)
        
        # Initial state
        x = np.zeros(dim)
        sigma = 2.0
        
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = x.copy()
        best_f = func(x)
        
        evals = 1
        gen = 0
        
        while evals < budget:
            if time.time() > self.deadline:
                break
                
            # Compute C_sqrt
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            # Generate population
            ys = np.array([x + sigma * C_sqrt @ rng.standard_normal(dim) for _ in range(lambda_pop)])
            
            # Evaluate
            fs = np.array([func(np.clip(y, -5, 5)) for y in ys])
            evals += len(fs)
            
            # Track best
            idx_best = np.argmin(fs)
            if fs[idx_best] < best_f:
                best_f = fs[idx_best]
                best_x = ys[idx_best].copy()
                
            if evals >= budget:
                break
                
            # Sort
            idx = np.argsort(fs)
            ys_sorted = ys[idx]
            fs_sorted = fs[idx]
            
            # Update mean
            x_new = np.dot(weights, ys_sorted[:mu])
            
            # Update path
            y_diff = (x_new - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_diff
            
            # Check h_sigma
            # Approximate the norm check
            norm_p_sigma = np.linalg.norm(p_sigma)
            # The denominator in the standard check is sqrt(1 - (1-c_sigma)^(2*(gen+1)))
            # For simplicity and robustness, we can use a simplified check or the standard one.
            # Standard:
            denom = np.sqrt(1 - (1 - c_sigma) ** (2 * (gen + 1)))
            h_sigma = 1 if (norm_p_sigma / denom < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1))) else 0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_diff
            
            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            
            # Rank-1 update
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c)
            
            # Rank-mu update
            for i in range(mu):
                diff = (ys_sorted[i] - x) / sigma
                C += c_mu * weights[i] * np.outer(diff, diff)
            
            # Update sigma
            sigma *= np.exp(c_sigma / 5 * (norm_p_sigma / np.sqrt(mu_eff) - 1))
            
            # Update x
            x = x_new
            gen += 1
            
        return best_x, best_f
# EVOLVE-BLOCK-END
