import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # Time management
        start_time = time.time()
        max_time = 290.0  # Safety margin for 300s limit
        
        # CMA-ES parameters
        lambda_ = 4  # Population size
        mu = max(1, lambda_ // 2)  # Number of parents
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        # Step size control parameters
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        # Initial state
        x_mean = np.zeros(n)
        sigma = 2.0
        pc = np.zeros(n)
        ps = np.zeros(n)
        C = np.eye(n)
        
        # Precompute matrix square root of C (identity initially)
        D = np.eye(n)
        # Cholesky of C for sampling
        C_chol = np.linalg.cholesky(C)
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls = 1
        
        while calls < budget:
            if time.time() - start_time > max_time:
                break
            
            # Generate population
            offspring = []
            offspring_f = []
            
            for i in range(lambda_):
                z = rng.standard_normal(n)
                y = x_mean + sigma * (D @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                offspring.append(y)
                offspring_f.append(fy)
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
                
                # Check time after each eval to avoid overshooting
                if calls >= budget:
                    break
            
            if calls >= budget:
                break
            
            # Sort offspring by fitness
            sorted_idx = np.argsort(offspring_f)
            selected = [offspring[i] for i in sorted_idx[:mu]]
            selected_f = [offspring_f[i] for i in sorted_idx[:mu]]
            
            # Compute weighted mean
            x_new = np.zeros(n)
            for i in range(mu):
                x_new += weights[i] * selected[i]
            
            # Update step size
            y_diff = x_new - x_mean
            # Compute ps
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (D @ y_diff)
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Update evolution path pc
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) / sigma * y_diff
            
            # Rank-mu update for covariance matrix
            C = (1 - c_1 - c_mu) * C
            C += c_1 * np.outer(pc, pc)
            for i in range(mu):
                z_i = D @ (selected[i] - x_mean) / sigma
                C += c_mu * weights[i] * np.outer(z_i, z_i)
            
            # Update x_mean
            x_mean = x_new
            x_mean = np.clip(x_mean, -5, 5)
            
            # Update D (matrix square root of C)
            # Use Cholesky decomposition
            try:
                C_chol = np.linalg.cholesky(C)
                D = C_chol
            except np.linalg.LinAlgError:
                # If C is not positive definite, regularize
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-10, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C_chol = np.linalg.cholesky(C)
                D = C_chol
        
        return best_x, best_f
