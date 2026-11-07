import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # CMA-ES Parameters
        lambda_pop = 4 + int(3 * np.log(dim))
        if lambda_pop < 4:
            lambda_pop = 4
        # For dim=5, lambda = 8
        
        mu = lambda_pop // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # State
        x_mean = np.zeros(dim)
        sigma = 2.0
        
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        
        def update_C_sqrt():
            nonlocal C, C_sqrt
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-16, None)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
            
        update_C_sqrt()
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        evals = 1
        gen = 0
        
        while evals < budget:
            if time.time() > deadline:
                break
                
            gen += 1
            # Generate population
            Z = rng.standard_normal((lambda_pop, dim))
            X = x_mean + sigma * (Z @ C_sqrt.T)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            evals += lambda_pop
            
            # Sort by fitness
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]
            
            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = X_sorted[0].copy()
                
            # Update mean
            x_new = np.sum(weights[:, np.newaxis] * X_sorted[:mu], axis=0)
            
            # Update evolution paths
            y = (x_new - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y
            
            # h_sigma condition
            # Approximate generation count for the exponential term
            gen_count = gen
            denom = np.sqrt(1 - (1 - c_sigma)**(2 * gen_count))
            if denom == 0:
                denom = 1e-16
            h_sigma = (np.linalg.norm(p_sigma) / denom < (1 + 2 / dim) * np.sqrt((mu_eff + 1) / (dim + 1)))
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y
            
            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            # y_w is the weighted mean of y
            # The term is sum w_i (y_i - y_w)(y_i - y_w)^T
            # We have X_sorted[:mu] - x_mean = sigma * y_i
            # So (X_sorted[:mu] - x_mean) / sigma = y_i
            Y_sorted = (X_sorted[:mu] - x_mean) / sigma
            Y_w = np.sum(weights[:, np.newaxis] * Y_sorted, axis=0)
            Y_diff = Y_sorted - Y_w
            cov_update = np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(Y_diff, Y_diff), axis=0)
            
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * cov_update
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update step size (CSA)
            sigma = sigma * np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Clip sigma
            sigma = np.clip(sigma, 1e-5, 10.0)
            
            # Update mean
            x_mean = x_new
            
            update_C_sqrt()
            
        return best_x, best_f
