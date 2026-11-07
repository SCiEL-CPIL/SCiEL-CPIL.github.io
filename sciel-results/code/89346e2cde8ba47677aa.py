"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for continuous black-box functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size lambda
        lam = int(4 + 3 * np.log(dim))
        
        # Initial parameters
        x0 = np.zeros(dim)
        sigma0 = 2.0
        sigma = sigma0
        
        # Initial covariance matrix (identity)
        C = np.eye(dim)
        # Initial inverse covariance matrix
        C_inv = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Time constants
        mu = lam // 2  # Number of recombination weights
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Accumulators for Hessian update
        H = np.zeros((dim, dim))
        
        # Best solution so far
        x_best = x0.copy()
        f_best = func(x_best)
        
        calls = 1
        
        # Main loop
        while calls < budget:
            # Generate offspring
            Y = []
            for i in range(lam):
                # Sample from N(0, C)
                # Use Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback if C is not positive definite due to numerical issues
                    C = 0.5 * (C + C.T) + 1e-6 * np.eye(dim)
                    L = np.linalg.cholesky(C)
                
                z = np.random.randn(dim)
                y = L @ z
                # Clip to box
                x_off = np.clip(x0 + sigma * y, -5, 5)
                Y.append(x_off)
            
            # Evaluate offspring
            F = []
            for x_off in Y:
                f_val = func(x_off)
                F.append(f_val)
                calls += 1
                if calls >= budget:
                    break
            
            if calls >= budget:
                break
                
            F = np.array(F)
            
            # Sort by fitness
            sort_idx = np.argsort(F)
            Y_sorted = np.array([Y[i] for i in sort_idx])
            F_sorted = F[sort_idx]
            
            # Update best solution
            if F_sorted[0] < f_best:
                f_best = F_sorted[0]
                x_best = Y_sorted[0].copy()
                
            # Update mean
            x0 = np.sum(w[:, np.newaxis] * Y_sorted[:mu], axis=0)
            
            # Update evolution paths
            # Need C_inv_sqrt for p_sigma
            # Compute eigen decomposition for C
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip eigenvalues to avoid negative
            eigvals = np.clip(eigvals, 1e-16, None)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            y_vec = np.sum(w[:, np.newaxis] * (Y_sorted[:mu] - x0) / sigma, axis=0)
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt @ y_vec
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_vec
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Update covariance matrix
            # H = sum(w_i * (y_i y_i^T) - C)
            Y_norm = (Y_sorted[:mu] - x0) / sigma
            H = np.sum(w[:, np.newaxis, np.newaxis] * Y_norm[:, :, np.newaxis] * Y_norm[:, np.newaxis, :], axis=0)
            H += (1 + 2 * c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1)) * np.outer(p_sigma, p_sigma) - c_sigma * (2 - c_c) * np.outer(p_c, p_c)
            
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c p_c^T + (1 - h_sigma) * C) + c_mu * H
            # c_1 and c_mu
            c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1/mu_eff) / ((dim + 2)**2 + mu_eff))
            
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - 1) * C) + c_mu * H
            
            # Symmetrize C
            C = 0.5 * (C + C.T)
            
            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            if np.min(eigvals) < 1e-16:
                C = eigvecs @ np.diag(np.clip(eigvals, 1e-16, None)) @ eigvecs.T

        return x_best, f_best
# EVOLVE-BLOCK-END
