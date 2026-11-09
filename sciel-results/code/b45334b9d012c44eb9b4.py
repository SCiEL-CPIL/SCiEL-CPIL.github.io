"""Optimized solver for BBOB f18 (Schaffers F7 ill-conditioned) using CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """
    Solver using CMA-ES (Covariance Matrix Adaptation Evolution Strategy).
    
    CMA-ES is the state-of-the-art algorithm for black-box continuous optimization.
    It adapts the covariance matrix of the sampling distribution to the shape of
    the fitness landscape, making it highly effective for ill-conditioned problems
    like Schaffers F7.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin before 330s kill
        
        calls = 0
        best_f = np.inf
        best_x = np.zeros(dim)
        
        rng = np.random.default_rng(42)

        def counting_func(x):
            nonlocal calls, best_f, best_x
            if calls >= budget:
                return best_f
            calls += 1
            x_clipped = np.clip(x, -5, 5)
            fx = func(x_clipped)
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # --- CMA-ES Implementation ---
        # Population size
        lambda_ = 4 + int(3 * np.log(dim))  # Standard CMA-ES population size
        mu = lambda_ // 2  # Number of parents
        
        # Weights for recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Variance-effective population size
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Step size adaptation parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Mean adaptation parameter
        c_mean = (mu_eff + 2) / (dim + mu_eff + 5)
        
        # Covariance adaptation parameters
        c_cov = (2 + 2 * mu_eff / dim) / (dim ** 2 + dim + 2 + 2 * mu_eff / dim)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        
        # Initial state
        x_mean = np.zeros(dim)
        sigma = 2.0  # Initial step size for [-5, 5]^5
        C = np.eye(dim)  # Covariance matrix
        pc = np.zeros(dim)  # Evolution path for covariance
        ps = np.zeros(dim)  # Evolution path for step size
        
        # Precompute Cholesky factor
        B = np.eye(dim)  # Eigenvectors
        D = np.ones(dim)  # Eigenvalues (square roots)
        
        # Initial exploration: evaluate a few random points to get a better starting point
        n_init = min(10, budget // 5)
        for _ in range(n_init):
            if calls >= budget or time.time() > deadline:
                break
            x_init = rng.uniform(-5, 5, dim)
            f_init = counting_func(x_init)
        
        # Start CMA-ES from the best initial point
        x_mean = best_x.copy()
        
        # Main CMA-ES loop
        while calls < budget and time.time() < deadline:
            # Generate population
            xs = np.zeros((lambda_, dim))
            fs = np.zeros(lambda_)
            
            for i in range(lambda_):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Sample from multivariate normal
                z = rng.standard_normal(dim)
                x_i = x_mean + sigma * (B @ (D * z))
                x_i = np.clip(x_i, -5, 5)
                
                fs[i] = counting_func(x_i)
                xs[i] = x_i
            
            if calls >= budget:
                break
            
            # Sort by fitness
            order = np.argsort(fs)
            xs = xs[order]
            fs = fs[order]
            
            # Update mean
            x_new_mean = np.sum(weights[:, None] * xs[:mu], axis=0)
            
            # Update evolution paths
            inv_sqrtC = np.linalg.inv(np.sqrtm(C))
            pc = (1 - c_sigma) * pc + c_sigma * d_sigma * inv_sqrtC @ (x_new_mean - x_mean)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * inv_sqrtC @ (x_new_mean - x_mean) / sigma
            
            # Update covariance matrix
            delta_heav = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            C = ((1 + c_cov * delta_heav - c_cov) * C + 
                 c_cov * (np.outer(pc, pc) / (1 - (1 - c_sigma) ** 2) - 
                          C * (1 - (1 - c_sigma) ** 2) / (1 - (1 - c_sigma) ** 2) + 
                          np.sum(weights[:, None, None] * np.outer(xs[:mu] - x_mean, xs[:mu] - x_mean), axis=0) / sigma ** 2))
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update step size
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** 2) - np.sqrt(dim * (1 - np.exp(-2 * c_sigma)))))
            
            # Update mean
            x_mean = x_new_mean
            
            # Ensure C is positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            B, D = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(np.diag(D), 1e-20))
            B = eigvecs

        return best_x, best_f
# EVOLVE-BLOCK-END
