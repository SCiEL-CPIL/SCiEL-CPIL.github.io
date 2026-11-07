import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        # CMA-ES Parameters
        # Population size
        lambda_ = 6 + int(3 * np.log(dim))
        mu = lambda_ // 2
        
        # Initial mean
        x_mean = np.zeros(dim)
        
        # Initial step size
        # Large sigma to ensure early coverage and progress
        sigma = 2.0
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Path variables for sigma adaptation
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Adaptation parameters
        c_sigma = (mu + 2) / (dim + mu + 5)
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        d_sigma = 1 + 2 * max(0, (math.sqrt(dim) - 1) / (dim + 3) - 1) + c_sigma
        c_cov = (2 + 2 * mu / dim) / (dim + 1 + 2 * mu / dim)
        
        # Precompute eigen decomposition for sampling
        # We will update C every generation
        
        best_x = x_mean.copy()
        best_f = func(x_mean)
        
        calls_used = 1
        remaining_budget = budget - calls_used
        
        while remaining_budget > 0:
            # Check time
            if time.time() > deadline:
                break
            
            # Determine how many candidates we can evaluate in this generation
            # We want to use up the budget in chunks of lambda_, but if remaining < lambda_, we evaluate remaining
            n_evals = min(lambda_, remaining_budget)
            
            # Sample candidates
            # We need to sample from N(x_mean, sigma^2 * C)
            # Use Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset or fix
                C = np.eye(dim)
                L = np.eye(dim)
            
            candidates = []
            fvals = []
            
            for i in range(n_evals):
                # Standard normal
                z = np.random.randn(dim)
                # Transform
                x_new = x_mean + sigma * (L @ z)
                # Clip to bounds
                x_new = np.clip(x_new, -5, 5)
                
                # Evaluate
                f_new = func(x_new)
                calls_used += 1
                remaining_budget -= 1
                
                candidates.append(x_new)
                fvals.append(f_new)
                
                if f_new < best_f:
                    best_f = f_new
                    best_x = x_new.copy()
            
            # Sort candidates by fitness
            sorted_indices = np.argsort(fvals)
            
            # Update mean
            # Use the best mu candidates
            w = np.ones(mu) / mu  # Uniform weights for simplicity, or standard CMA-ES weights
            # Standard CMA-ES weights:
            # w_i = log(mu/2 + 1) - log(i) for i=1..mu
            # Let's use standard weights for better performance
            weights = np.array([np.log(mu/2 + 1) - np.log(i+1) for i in range(mu)])
            weights = weights / np.sum(weights)
            
            # x_new_mean = sum(w_i * x_i) for i in top mu
            x_new_mean = np.zeros(dim)
            for i in range(mu):
                idx = sorted_indices[i]
                x_new_mean += weights[i] * candidates[idx]
            
            # Update covariance
            # C = (1 - c_cov * (1 + (1 - nu) * c_sigma * (1 - c_sigma))) * C 
            #    + c_cov * (nu * p_sigma * p_sigma.T + sum(w_i * y_i * y_i.T))
            # where y_i = (x_i - x_mean) / sigma
            
            y_vecs = [(candidates[sorted_indices[i]] - x_mean) / sigma for i in range(mu)]
            
            # Rank-one update
            rank_one = np.zeros((dim, dim))
            for i in range(mu):
                rank_one += weights[i] * np.outer(y_vecs[i], y_vecs[i])
            
            # Rank-mu update for covariance (simplified, using only top mu)
            # Note: Full CMA-ES uses all lambda for rank-mu, but top mu is often sufficient and faster
            # Actually, standard CMA-ES uses the top mu for the mean and the covariance update uses the top mu as well in the rank-mu part?
            # No, standard CMA-ES covariance update:
            # C = (1 - c_cov * (1 + (1 - nu) * c_sigma * (1 - c_sigma))) * C + c_cov * (nu * p_sigma * p_sigma.T + sum_{i=1}^mu w_i * y_i * y_i.T)
            # where nu = sum(w_i^2)
            
            nu = np.sum(weights**2)
            
            # Update p_sigma
            # p_sigma = (1 - c_sigma) * p_sigma + c_sigma * sqrt(nu) * (x_new_mean - x_mean) / (sigma * ||C^{1/2}||_F)
            # Approximation for ||C^{1/2}||_F is sqrt(trace(C))
            c_norm = np.sqrt(np.trace(C))
            if c_norm == 0:
                c_norm = 1.0
            
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(nu) * (x_new_mean - x_mean) / (sigma * c_norm)
            
            # Update sigma
            # sigma = sigma * exp(c_sigma / d_sigma * (||p_sigma|| / E||N(0, I)|| - 1))
            # E||N(0, I)|| = sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim^2))
            e_norm = math.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / e_norm - 1))
            
            # Update p_c
            # p_c = (1 - c_c) * p_c + c_c * sqrt(nu) * (x_new_mean - x_mean) / sigma
            p_c = (1 - c_c) * p_c + c_c * np.sqrt(nu) * (x_new_mean - x_mean) / sigma
            
            # Update C
            # C = (1 - c_cov * (1 + (1 - nu) * c_sigma * (1 - c_sigma))) * C + c_cov * (nu * p_sigma * p_sigma.T + rank_one)
            c_cov_eff = c_cov * (1 + (1 - nu) * c_sigma * (1 - c_sigma))
            C = (1 - c_cov_eff) * C + c_cov_eff * (nu * np.outer(p_sigma, p_sigma) + rank_one)
            
            # Symmetrize C
            C = (C + C.T) / 2
            
            # Update mean
            x_mean = x_new_mean
            
            # Ensure mean is within bounds (CMA-ES doesn't natively handle bounds, so we clip)
            # Clipping the mean can be problematic, but it's better than letting it drift out
            x_mean = np.clip(x_mean, -5, 5)
            
            # If sigma becomes too small, we might stagnate. 
            # But with 500 calls, we should be fine.
            
        return best_x, best_f
