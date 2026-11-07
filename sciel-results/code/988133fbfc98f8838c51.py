"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES based optimizer for BBOB f4."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 290.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size lambda
        lam = max(4, int(4 + 3 * np.log(dim)))
        
        # Initial sigma
        # The search space is [-5, 5]^5. A good initial sigma is roughly 1/5 of the range.
        sigma = 1.5
        
        # Mean (center of the distribution)
        # Start at the center of the box
        x_mean = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Inverse square root of C
        D = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Adaptation parameters
        # Standard CMA-ES updates
        mu = lam // 2  # Number of parents
        weights = np.zeros(mu)
        # Logarithmic weights
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Step size control
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Covariance matrix adaptation
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_cov = (2 + 2 * mu_eff / dim) / (dim + 1 + 3 * c_c)
        
        # Chi-square approximation for normalization
        chi_lambda = np.sqrt(lam * (1 - 1/(4*dim) - 1/(16*dim**2)))
        
        # Best solution found so far
        x_best = np.full(dim, 5.0) # Initialize to max possible value (approx)
        f_best = float('inf')
        
        calls = 0
        
        # Initial evaluation of the mean
        x_init = np.clip(x_mean, -5, 5)
        f_init = func(x_init)
        calls += 1
        if f_init < f_best:
            f_best = f_init
            x_best = x_init.copy()
            
        # If we already used too many calls (shouldn't happen with budget>=1)
        if calls >= budget:
            return x_best, f_best

        # Main loop
        while calls < budget:
            # Time check
            if time.time() - start_time > time_limit:
                break
                
            # Generate offspring
            ys = []
            fs = []
            
            for i in range(lam):
                # Check budget inside the loop to avoid overshooting
                if calls >= budget:
                    break
                
                # Sample
                # z ~ N(0, I)
                z = np.random.randn(dim)
                # y = x_mean + sigma * D @ z
                y = x_mean + sigma * (D @ z)
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls += 1
                
                ys.append(y)
                fs.append(fy)
                
                # Update global best
                if fy < f_best:
                    f_best = fy
                    x_best = y.copy()
            
            if not ys:
                break
                
            ys = np.array(ys)
            fs = np.array(fs)
            
            # Sort by fitness
            idx = np.argsort(fs)
            ys = ys[idx]
            fs = fs[idx]
            
            # Take the best mu individuals
            ys_parents = ys[:mu]
            
            # Weighted mean of parents
            x_new_mean = np.sum(weights[:, None] * ys_parents, axis=0)
            
            # Update evolution path for covariance
            y_mean = x_new_mean - x_mean
            y_c = y_mean / (sigma * np.sqrt(np.sum((D @ D.T)))) # Approximation for normalization, 
            # Actually, standard CMA-ES uses:
            # y_c = (x_new_mean - x_mean) / (sigma * ||D||_F) is not quite right.
            # Let's use the standard formula:
            # y_c = (x_new_mean - x_mean) / (sigma * np.sqrt(np.trace(D @ D.T)))
            # Wait, the standard update is:
            # p_c = (1 - c_c) * p_c + c_c * sqrt(mu_eff) / (sigma * ||D||) * (x_new_mean - x_mean)
            # where ||D|| is the Frobenius norm? No, it's related to the expected norm.
            # Let's stick to a robust implementation.
            
            # Recalculate D as inverse square root of C
            # For efficiency in this short run, we can use SVD or eigendecomp
            # C is symmetric positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip small negative eigenvalues due to numerical errors
            eigvals = np.maximum(eigvals, 1e-12)
            D = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Update p_c
            # The term (x_new_mean - x_mean) / (sigma * ||D||) needs care.
            # Standard CMA-ES:
            # p_c = (1 - c_c) p_c + c_c * sqrt(mu_eff) * (x_new_mean - x_mean) / (sigma * ||D||_F) 
            # is NOT correct. The denominator is the expected norm of the standard normal transformed by D?
            # Actually, the standard formula is:
            # p_c = (1 - c_c) p_c + c_c * sqrt(mu_eff) * (x_new_mean - x_mean) / (sigma * ||D||_F)
            # where ||D||_F is the Frobenius norm of D.
            # Let's verify. In Hansen's paper, it's:
            # p_c = (1-c_c)p_c + c_c * sqrt(mu_eff) * (x_new_mean - x_mean) / (sigma * ||D||_F)
            # Yes, ||D||_F is the Frobenius norm.
            
            norm_D = np.linalg.norm(D, 'fro')
            if norm_D > 0:
                p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * (x_new_mean - x_mean) / (sigma * norm_D)
            else:
                p_c = (1 - c_c) * p_c
            
            # Update p_sigma
            # p_sigma = (1 - c_sigma) p_sigma + c_sigma * sqrt(mu_eff) * (x_new_mean - x_mean) / (sigma * ||D||_F)
            # Wait, for p_sigma, it uses the same normalization?
            # Yes, p_sigma is updated with the same weighted mean shift.
            if norm_D > 0:
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * (x_new_mean - x_mean) / (sigma * norm_D)
            else:
                p_sigma = (1 - c_sigma) * p_sigma
            
            # Update sigma
            # sigma = sigma * exp( c_sigma / d_sigma * (||p_sigma|| / chi_lambda - 1) )
            norm_p_sigma = np.linalg.norm(p_sigma)
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_lambda - 1))
            
            # Update C
            # C = (1 - c_cov - mu_eff * c_cov) C + c_cov * (p_c p_c^T + 1/sigma^2 sum w_i (y_i - x_mean)(y_i - x_mean)^T)
            # Note: y_i here are the original samples, not the parents?
            # No, the sum is over the parents.
            # The term is: (x_new_mean - x_mean)(x_new_mean - x_mean)^T is NOT included directly, 
            # it's part of the second term via the sum?
            # Actually, the standard update is:
            # C = (1 - c_cov - mu_eff*c_cov)C + c_cov * (p_c p_c^T + 1/sigma^2 * sum_{i=1}^mu w_i (y_i - x_mean)(y_i - x_mean)^T)
            # where y_i are the selected parents.
            
            # Calculate the sum of outer products
            # y_i - x_mean
            diff = ys_parents - x_mean
            # Weighted sum of outer products
            # w_i is weights
            # term = sum w_i * diff_i * diff_i^T
            # We can compute this as: (diff.T @ (weights * diff)) / sigma^2 ?
            # No: sum w_i * v_i v_i^T = V^T diag(w) V
            # Let V be the matrix of rows (y_i - x_mean)
            # V = diff
            # term = (diff.T * weights) @ diff / sigma^2
            # Wait, diff.T @ (weights[:, None] * diff)
            
            weighted_diff = weights[:, None] * diff
            sum_outer = diff.T @ weighted_diff
            sum_outer /= (sigma * sigma)
            
            # p_c p_c^T
            outer_pc = np.outer(p_c, p_c)
            
            # Update C
            C = (1 - c_cov - mu_eff * c_cov) * C + c_cov * (outer_pc + sum_outer)
            
            # Ensure C is symmetric
            C = (C + C.T) / 2.0
            
            # Update mean
            x_mean = x_new_mean
            
            # Optional: Check if sigma is too small or too large
            if sigma < 1e-12:
                break
            if sigma > 100:
                sigma = 100

        return x_best, f_best
# EVOLVE-BLOCK-END
