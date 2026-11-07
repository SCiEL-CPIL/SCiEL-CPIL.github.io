import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        N = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lambda_ = 4 + int(3 * np.log(N))  # Population size
        if lambda_ < 4: lambda_ = 4
        mu = lambda_ // 2  # Number of parents
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Adaptation parameters
        c_sigma = (mu_eff + 2) / (lambda_ + mu_eff + 5)
        c_c = (4 + mu_eff / lambda_) / (lambda_ + 4 + 2 * mu_eff / lambda_)
        c_1 = 2.0 / (lambda_**2 + 6)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1/mu_eff) / (lambda_**2 + 4))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (lambda_ + 1)) - 1) + c_sigma
        
        # Initial state
        x_mean = np.zeros(N)
        sigma = 2.0
        C = np.eye(N)
        
        # Evolution paths
        p_sigma = np.zeros(N)
        p_c = np.zeros(N)
        
        # Cholesky factor
        try:
            Ch = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            Ch = np.eye(N)
            
        # Tracking best
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls_made = 1
        
        # Time limit guard
        start_time = time.time()
        time_limit = 250  # Safety margin for 300s limit
        
        rng = np.random.default_rng(0)

        while calls_made < budget:
            if time.time() - start_time > time_limit:
                break
                
            # Generate population
            Y = np.zeros((lambda_, N))
            for i in range(lambda_):
                z = rng.standard_normal(N)
                Y[i] = x_mean + sigma * (Ch @ z)
                # Clip to box
                Y[i] = np.clip(Y[i], -5, 5)
                
            # Evaluate population
            F = np.array([func(Y[i]) for i in range(lambda_)])
            calls_made += lambda_
            
            # Sort
            idx = np.argsort(F)
            F_sorted = F[idx]
            Y_sorted = Y[idx]
            
            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = Y_sorted[0].copy()
                
            # Update mean
            x_old = x_mean.copy()
            x_mean = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)
            
            # Update sigma
            y_sigma = (x_mean - x_old) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(N) - 1))
            
            # Update covariance matrix
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean - x_old) / sigma
            
            # Update C
            # Rank-1 update
            norm_p_sigma = np.linalg.norm(p_sigma)
            denom = np.sqrt(1 - (1 - c_sigma)**2)
            if denom == 0:
                h_sigma = 1
            else:
                h_sigma = 1 if norm_p_sigma / denom < (1.4 + 2/(N+1)) * np.sqrt(N+1) else 0
            
            C = (1 + c_1 * h_sigma * (c_c * (2 - c_c) - 1) - c_1) * C + \
                c_1 * (1 + h_sigma * (c_c * (2 - c_c) - 1)) * np.outer(p_c, p_c)
                
            # Rank-mu update
            delta = np.zeros(N)
            for i in range(mu):
                y_i = (Y_sorted[i] - x_old) / sigma
                delta += weights[i] * np.outer(y_i, y_i)
            C = C + c_mu * (np.outer(p_c, p_c) * (h_sigma - 1) - delta)
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update Cholesky
            try:
                Ch = np.linalg.cholesky(C + 1e-10 * np.eye(N))
            except np.linalg.LinAlgError:
                # If Cholesky fails, regularize more
                C = C + 0.1 * np.eye(N)
                try:
                    Ch = np.linalg.cholesky(C)
                except:
                    Ch = np.eye(N)
                    C = np.eye(N)
                    
            # Clamp sigma to prevent explosion or degeneration
            sigma = np.clip(sigma, 1e-10, 10.0)

        return best_x, best_f
