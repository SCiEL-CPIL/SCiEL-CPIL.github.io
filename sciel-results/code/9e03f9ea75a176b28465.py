import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w**2)
        
        cs = (mueff + 1.5 + 2.0 / (n + 1.0)) / (n + mueff + 5.0)
        ds = 1 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (n + 1.0)) - 1.0) + cs
        
        c1 = (2.0 / ((n + 1.8)**2 + mueff)) * (2.0 - (mueff + 1.0) / ((n + 1.8)**2 + mueff))
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((n + 2.0)**2 + mueff))
        
        # Initial state
        x_mean = np.zeros(n)
        sigma = 2.0
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        C = np.eye(n)
        
        # Pre-compute
        def update_C_sqrt(C):
            w_eig, V = np.linalg.eigh(C)
            w_eig = np.clip(w_eig, 1e-12, None)
            C_sqrt = V @ np.diag(np.sqrt(w_eig)) @ V.T
            C_inv = V @ np.diag(1.0 / w_eig) @ V.T
            return C_sqrt, C_inv, w_eig

        C_sqrt, C_inv, w_eig = update_C_sqrt(C)
        
        evals = 0
        best_x = x_mean.copy()
        best_f = func(best_x)
        evals += 1
        
        start_time = time.time()
        timeout = 290.0
        
        rng = np.random.default_rng(42)
        
        while evals < budget:
            if time.time() - start_time > timeout:
                break
                
            # Generate population
            z = rng.standard_normal((lam, n))
            X = x_mean + sigma * (z @ C_sqrt.T)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            evals += lam
            
            # Update best
            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = X[idx_min].copy()
                
            # Sort
            idx_sorted = np.argsort(F)
            X_sorted = X[idx_sorted]
            F_sorted = F[idx_sorted]
            
            # Mean update
            y_w = np.sum(w[:mu, np.newaxis] * X_sorted[:mu], axis=0)
            x_mean_new = x_mean + mueff * (y_w - x_mean)
            
            # Path updates
            y_w_std = y_w / sigma
            p_sigma = (1.0 - cs) * p_sigma + cs * np.sqrt(mueff) * y_w_std
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1.0 - (1.0 - cs)**(2 * (evals // lam + 1)))) < (1.4 + 2.0 / (n + 1.0)) * np.sqrt((mueff + 1.0) / (n + 1.0))
            
            p_c = (1.0 - c1) * p_c + h_sigma * np.sqrt(c1 * (2.0 - c1) * mueff) * y_w_std
            
            # Covariance update
            C = (1.0 - mu * c1 - cmu) * C + c1 * np.outer(p_c, p_c)
            
            # Rank-mu update
            y_sorted_std = X_sorted[:mu] / sigma
            C += cmu * np.sum(w[:mu, np.newaxis, np.newaxis] * y_sorted_std[:, np.newaxis, :] * y_sorted_std[:, np.newaxis, :].T, axis=0)
            
            x_mean = x_mean_new
            C_sqrt, C_inv, w_eig = update_C_sqrt(C)
            
            # Sigma update
            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / np.sqrt(mueff) - 1.0))
            
            # Convergence check
            if sigma < 1e-12 or np.max(w_eig) / np.min(w_eig) > 1e14:
                break
                
        return best_x, best_f
