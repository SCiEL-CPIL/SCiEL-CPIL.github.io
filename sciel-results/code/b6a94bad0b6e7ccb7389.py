import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin
        
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(n))
        
        # Weights
        mu = lam // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Strategy parameters
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        # Initial state
        x = np.zeros(n)
        sigma = 1.0
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Initial function evaluation
        fx = func(x)
        best_x = x.copy()
        best_f = fx
        calls = 1
        
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        def get_chol(C):
            try:
                L = np.linalg.cholesky(C)
                return L
            except np.linalg.LinAlgError:
                w, v = np.linalg.eigh(C)
                w = np.clip(w, 1e-12, None)
                return v @ np.diag(np.sqrt(w)) @ v.T

        C_sqrt = get_chol(C)
        C_sqrt_inv = np.linalg.inv(C_sqrt)
        
        for gen in range(budget - 1):
            if time.time() > deadline:
                break
                
            # Generate population
            rng = np.random.default_rng(42 + gen) # Seeded for reproducibility in debug, but random is fine
            # Actually, using a single rng instance is better for performance
            # Let's create rng outside loop
            pass # placeholder

        # Re-initialize rng outside loop for efficiency
        rng = np.random.default_rng(0)
        
        for gen in range(budget - 1):
            if time.time() > deadline:
                break
                
            Y = rng.standard_normal((lam, n))
            X = x[None, :] + sigma * (Y @ C_sqrt.T)
            X = np.clip(X, -5, 5)
            
            F = np.array([func(xi) for xi in X])
            calls += lam
            
            idx_best = np.argmin(F)
            if F[idx_best] < best_f:
                best_f = F[idx_best]
                best_x = X[idx_best].copy()
                
            idx_sort = np.argsort(F)
            X_sorted = X[idx_sort]
            
            # Update mean
            x_new = np.sum(weights[:, None] * X_sorted[:mu], axis=0)
            
            # Update step size
            y_sigma = (x_new - x) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mueff) * (C_sqrt_inv @ y_sigma)
            
            # Hessian correction
            h_sig = 0.0
            norm_p_sigma = np.linalg.norm(p_sigma)
            denom = np.sqrt(1 - (1 - c_sigma)**(2 * (calls + 1)))
            if denom > 1e-12:
                if norm_p_sigma / denom < (1.4 + 2 / (n + 1)) * chi_n:
                    h_sig = 1.0
            
            # Update path
            p_c = (1 - c_c) * p_c + h_sig * np.sqrt(c_c * (2 - c_c) * mueff) * (x_new - x) / sigma
            
            # Update covariance
            y_diffs = (X_sorted[:mu] - x[None, :]) / sigma
            weighted_sum = (y_diffs.T * weights) @ y_diffs
            
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sig) * c_c * (2 - c_c) * C) + c_mu * weighted_sum
            C = (C + C.T) / 2
            
            # Update Cholesky
            C_sqrt = get_chol(C)
            C_sqrt_inv = np.linalg.inv(C_sqrt)
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            
            x = x_new
            
            if calls >= budget:
                break
                
        return best_x, best_f
