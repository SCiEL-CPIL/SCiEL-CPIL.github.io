"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES for the Bent Cigar function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 330s limit
        
        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(self.dim))
        sigma = 2.0
        mean = np.zeros(self.dim)
        C = np.eye(self.dim)
        C_sqrt_inv = np.eye(self.dim)
        
        p_sigma = np.zeros(self.dim)
        p_c = np.zeros(self.dim)
        
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (self.dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (self.dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / self.dim) / (self.dim + 4 + 2 * mu_eff / self.dim)
        c_1 = 2 / ((self.dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((self.dim + 2) ** 2 + mu_eff))
        
        best_x = mean.copy()
        best_f = func(best_x)
        calls_made = 1
        
        # Pre-compute Cholesky factor for initial C (I)
        # We will update C and its inverse square root each step
        
        while calls_made + lambda_ <= self.budget and time.time() < deadline:
            # Generate population
            pop = np.zeros((lambda_, self.dim))
            for i in range(lambda_):
                z = np.random.standard_normal(self.dim)
                # y = mean + sigma * C_sqrt_inv @ z
                pop[i] = mean + sigma * (C_sqrt_inv @ z)
                # Clip to box
                pop[i] = np.clip(pop[i], -5, 5)
            
            # Evaluate population
            f_vals = np.array([func(y) for y in pop])
            calls_made += lambda_
            
            # Update best
            min_idx = np.argmin(f_vals)
            if f_vals[min_idx] < best_f:
                best_f = f_vals[min_idx]
                best_x = pop[min_idx].copy()
            
            # Sort population
            order = np.argsort(f_vals)
            y_sorted = pop[order]
            
            # Weighted mean of best mu individuals
            y_w = np.zeros(self.dim)
            for i in range(mu):
                y_w += weights[i] * y_sorted[i]
            y_w /= np.sum(weights[:mu])
            
            # Save old mean for step size update
            old_mean = mean.copy()
            
            # Update mean
            mean = y_w
            
            # Update step size (sigma)
            # z_sigma = C_sqrt_inv @ (y_w - old_mean) / sigma
            z_sigma = C_sqrt_inv @ (y_w - old_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_sigma
            
            h_sigma = 0
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls_made - 1))) < (1.4 + 2 / (self.dim + 1)) * np.trace(C_sqrt_inv) / self.dim:
                h_sigma = 1
            
            # Update C
            # p_c = (1 - c_c) * p_c + h_sigma * sqrt(c_c * (2 - c_c) * mu_eff) * (y_w - old_mean) / sigma
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_w - old_mean) / sigma
            
            rank_1 = np.outer(p_c, p_c)
            rank_mu = np.zeros((self.dim, self.dim))
            for i in range(mu):
                y_i = (y_sorted[i] - old_mean) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)
            
            C = (1 + c_1 * h_sigma * (self.dim + 1) - c_1 - c_mu) * C + c_1 * (self.dim + 2 - h_sigma * (self.dim + 1)) * rank_1 + c_mu * rank_mu
            
            # Update C_sqrt_inv
            # Efficient update of inverse square root is complex. 
            # For dim=5, we can just recompute Cholesky of C^-1 or use eigen decomposition.
            # Since dim is small, recomputing C_sqrt_inv via eigen decomposition of C is safe and simple.
            # C is symmetric positive definite.
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            # Clip eigenvalues to avoid numerical issues
            eigenvalues = np.maximum(eigenvalues, 1e-12)
            C_sqrt_inv = (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T
            
            # Ensure C_sqrt_inv is symmetric
            C_sqrt_inv = (C_sqrt_inv + C_sqrt_inv.T) / 2.0
            
            # Optional: Restart mechanism if sigma becomes too small or large
            if sigma < 1e-10 or sigma > 1e10:
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
