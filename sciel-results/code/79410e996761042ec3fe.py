"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES for multimodal optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size lambda
        lambda_ = int(4 + 3 * np.log(dim))
        if lambda_ < 4:
            lambda_ = 4
            
        # Weights for selection
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Variance-effectiveness
        mueff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        
        # Initial step size
        sigma = 2.0
        # Initial mean (center of search space)
        x_mean = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        # Cholesky factor
        C_sqrt = np.eye(dim)
        
        # Path accumulators
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Precompute weights for sampling
        # We need the top 'mu' points to update the mean and covariance
        
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        # Helper to compute C_sqrt from C
        def update_chol(C):
            # Use SVD or Cholesky; SVD is more robust for near-singular
            U, s, Vt = np.linalg.svd(C)
            # Ensure positive definite
            s = np.maximum(s, 1e-12)
            return U @ np.diag(np.sqrt(s)) @ Vt

        C_sqrt = update_chol(C)
        
        while calls_made < budget:
            # Generate population
            y = rng.standard_normal((lambda_, dim))
            x = np.array([x_mean + sigma * (y[i] @ C_sqrt) for i in range(lambda_)])
            
            # Evaluate
            fvals = np.array([func(x[i]) for i in range(lambda_)])
            calls_made += lambda_
            
            # Update best
            for i in range(lambda_):
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = x[i].copy()
            
            # Sort by fitness
            order = np.argsort(fvals)
            y_sorted = y[order]
            x_sorted = x[order]
            fvals_sorted = fvals[order]
            
            # Update mean
            x_old = x_mean.copy()
            x_mean = np.sum(weights[:, None] * x_sorted[:mu], axis=0)
            
            # Update path p_sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * (C_sqrt.T @ ((x_mean - x_old) / sigma))
            
            # Hessian norm of p_sigma
            chi_n = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (24 * dim**2))
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - cs)**(2 * (calls_made // lambda_ + 1)))) < (1.4 + 2. / (dim + 1)) * chi_n
            
            # Update path p_c
            p_c = (1 - cs) * p_c + h_sigma * np.sqrt(cs * (2 - cs) * mueff) * (C_sqrt.T @ ((x_mean - x_old) / sigma))
            
            # Update covariance matrix
            rank1_update = np.outer(p_c, p_c)
            rankmu_update = np.zeros((dim, dim))
            for i in range(mu):
                z = (x_sorted[i] - x_old) / sigma
                rankmu_update += weights[i] * np.outer(z, z)
                
            C = (1 + c1 * h_sigma * (1 - h_sigma) - c1 - cmu) * C + c1 * rank1_update + cmu * rankmu_update
            
            # Ensure positive definiteness
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp(cs / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update C_sqrt
            C_sqrt = update_chol(C)
            
            # Break if budget exhausted
            if calls_made >= budget:
                break
                
        return best_x, best_f
# EVOLVE-BLOCK-END
