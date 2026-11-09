import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin
        
        d = self.dim
        # Population size
        lambda_pop = 4 + int(3 * np.log(d))
        
        # Weights
        mu = lambda_pop // 2
        weights = np.log(lambda_pop/2 + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_w = np.sum(weights)**2 / np.sum(weights**2)
        
        # CMA-ES parameters
        c_sigma = (4 + lambda_pop/d) / (4*d + 4 + lambda_pop/d)
        d_sigma = 1 + 2 * max(0, np.sqrt((lambda_pop/2 - 1)/(d + 1)) - 1) + c_sigma
        c_1 = 2 / ((d + 1.3)**2 + lambda_pop)
        c_mu = min(1 - c_1, 2 * (mu_w - 2 + 1/mu_w) / ((d + 2)**2 + lambda_pop))
        
        # Initialization
        rng = np.random.default_rng(0)
        mean = rng.uniform(-5, 5, d)
        sigma = 2.0
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        cs = (np.sqrt((lambda_pop + 2) * c_mu) + 1) / (d + np.sqrt((lambda_pop + 2) * c_mu))
        ds = 1 + 2 * max(0, np.sqrt((mu_w - 1) / (d + 1)) - 1) + cs
        
        best_x = mean
        best_f = func(best_x)
        calls = 1
        
        while calls < self.budget:
            if time.time() > deadline:
                break
                
            # Sample
            B = rng.multivariate_normal(np.zeros(d), C, size=lambda_pop)
            # Actually, rng.multivariate_normal is slow. Use Cholesky or just standard normal and transform?
            # For small d, it's fine. But let's use standard normal and transform by C^0.5?
            # C is not easily decomposed in each step without eigen decomp.
            # Standard CMA-ES uses C^0.5.
            # For d=5, we can compute eigen decomposition each step or use a simpler sampling.
            # Let's use the standard method: y_i ~ N(0, I), x_i = mean + sigma * (C^0.5 y_i)
            
            # Compute C^0.5
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            # Sample lambda points
            Y = rng.standard_normal((lambda_pop, d))
            X = mean + sigma * (Y @ C_sqrt.T) # (lambda_pop, d)
            
            # Clip to box
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            calls += lambda_pop
            
            # Update best
            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = X[idx_min].copy()
                
            # Sort by fitness
            idx_sorted = np.argsort(F)
            X_sorted = X[idx_sorted]
            F_sorted = F[idx_sorted]
            
            # Weighted mean
            m_new = np.sum(weights[:, np.newaxis] * X_sorted[:mu], axis=0)
            
            # Update step size
            # ps = (1 - cs) ps + sqrt(cs (2 - cs) mu_w) / sigma * C^{-1/2} (m_new - mean)
            C_inv_sqrt = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_w) / sigma * (C_inv_sqrt @ (m_new - mean))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) - np.sqrt(d - 1)))
            
            # Update mean
            mean = m_new
            
            # Update covariance matrix
            # C = (1 - c_1 - c_mu) C + c_1 (pc pc^T + (1 - (1 - cs)^2) C) + c_mu sum w_i (y_i y_i^T)
            # where y_i = (x_i - mean_old) / sigma
            pc = (1 - cs) * pc + np.sqrt(cs * (2 - cs) * mu_w) * (m_new - mean) / sigma
            
            # Rank-one update
            rank_one = np.outer(pc, pc)
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) < (1.4 + 1/d) * np.sqrt(d - 1):
                rank_one += (1 - (1 - cs)**2) * C
            
            # Rank-mu update
            Y_inv = (X_sorted[:mu] - mean) / sigma
            rank_mu = np.sum(weights[:, np.newaxis, np.newaxis] * (Y_inv[:, :, np.newaxis] * Y_inv[:, np.newaxis, :]), axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # If C becomes ill-conditioned, restart or reset
            eigvals_check = np.linalg.eigvalsh(C)
            if np.any(eigvals_check < 1e-12):
                C = np.eye(d)
                sigma = 2.0
                mean = best_x.copy()
                pc = np.zeros(d)
                ps = np.zeros(d)
                
        return best_x, best_f
