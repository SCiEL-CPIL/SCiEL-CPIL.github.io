import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        lam = 10  # population size
        mu = lam // 2  # number of weights for recombination
        
        # Logarithmic weights
        weights = np.log(np.arange(mu, 0, -1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Strategy parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        # Initial step size - large enough to explore the space
        # The minimizer is in [-4,4]^5, so start with sigma ~ 2
        sigma = 2.0
        mean = np.zeros(n)
        
        # Covariance matrix
        C = np.eye(n)
        # Precompute sqrt(C) via eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(C)
        C_sqrt = eigvecs @ np.diag(np.sqrt(np.maximum(eigvals, 1e-20))) @ eigvecs.T
        C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(np.maximum(eigvals, 1e-20))) @ eigvecs.T
        
        # Path variables
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Evolution path norm for sigma update
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))
        
        t_start = time.time()
        deadline = t_start + 285.0
        
        # Best tracking
        best_val = np.inf
        best_pos = mean.copy()
        
        calls = 0
        
        while calls < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            y_list = []
            x_list = []
            f_list = []
            
            for i in range(lam):
                if time.time() > deadline:
                    break
                # Sample from N(0, C)
                z = np.random.randn(n)
                y = C_sqrt @ z
                x = mean + sigma * y
                # Clip to box (func also clips, but let's be safe)
                x = np.clip(x, -5, 5)
                
                f_val = func(x)
                calls += 1
                
                y_list.append(y)
                x_list.append(x)
                f_list.append(f_val)
                
                if f_val < best_val:
                    best_val = f_val
                    best_pos = x.copy()
            
            if len(f_list) == 0:
                break
            
            # Sort by fitness
            idx = np.argsort(f_list)
            y_sorted = np.array([y_list[i] for i in idx])
            x_sorted = np.array([x_list[i] for i in idx])
            f_sorted = np.array([f_list[i] for i in idx])
            
            # Recombination
            y_w = np.sum(weights[:, None] * y_sorted[:mu], axis=0)
            x_w = np.sum(weights[:, None] * x_sorted[:mu], axis=0)
            
            # Update mean
            mean = x_w
            
            # Update p_sigma
            C_sqrt_old = C_sqrt
            # Need to recompute C_sqrt_inv for the update
            # Actually, let's use the standard update
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv @ (y_w / sigma)
            
            # Update p_c
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls // lam + 1)))) < (1 + 2 / (n + 1)) * chi_n
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_w / sigma
            
            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.sum(weights[:, None] * np.outer(y_sorted[:mu], y_sorted[:mu]), axis=0)
            
            C = (1 + c_1 * (2 - c_1) - c_1 - c_mu) * C + c_1 * (rank_one + delta_h * np.eye(n)) + c_mu * rank_mu
            
            # Ensure positive definite
            eigvals_C, eigvecs_C = np.linalg.eigh(C)
            eigvals_C = np.maximum(eigvals_C, 1e-10)
            C = eigvecs_C @ np.diag(eigvals_C) @ eigvecs_C.T
            
            # Update C_sqrt
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Clip mean to stay in bounds
            mean = np.clip(mean, -5, 5)
        
        return best_pos, best_val
