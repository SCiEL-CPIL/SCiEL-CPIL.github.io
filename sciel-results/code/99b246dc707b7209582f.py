import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        D = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        lam = 4 + int(3 * np.log(D))
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1.0 / np.sum(w**2)
        
        # Control parameters
        c_sigma = (mueff + 2.0) / (D + mueff + 5.0)
        d_sigma = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) * (D + 1.0) / mueff) - 1.0) + c_sigma
        c_c = (4.0 + mueff / D) / (D + 4.0 + 2.0 * mueff / D)
        c_1 = 2.0 / (D**2 + 18.0)
        c_mu = min(1.0 - c_1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / (D**2 + 18.0))
        
        # Initialization
        rng = np.random.default_rng(0)
        x_mean = np.zeros(D)
        sigma = 2.0
        C = np.eye(D)
        
        p_sigma = np.zeros(D)
        p_c = np.zeros(D)
        
        # Cholesky factor
        Csqrt = np.eye(D)
        
        def update_chol(C):
            try:
                L = np.linalg.cholesky(C)
                return L
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C_reg = eigvecs @ np.diag(eigvals) @ eigvecs.T
                L = np.linalg.cholesky(C_reg)
                return L
        
        Csqrt = update_chol(C)
        
        best_x = x_mean.copy()
        best_f = func(best_x)
        calls_made = 1
        
        start_time = time.time()
        deadline = 300.0
        safety_margin = 5.0
        
        gen = 0
        while calls_made + lam <= budget:
            # Check time
            if time.time() > deadline - safety_margin:
                break
            
            # Sample offspring
            z = rng.standard_normal((lam, D))
            y = z @ Csqrt.T
            x_offspring = x_mean + sigma * y.T
            x_offspring = np.clip(x_offspring, -5, 5)
            
            f_offspring = np.array([func(xi) for xi in x_offspring])
            calls_made += lam
            gen += 1
            
            # Update best
            min_idx = np.argmin(f_offspring)
            if f_offspring[min_idx] < best_f:
                best_f = f_offspring[min_idx]
                best_x = x_offspring[min_idx].copy()
                
            # Sort offspring
            order = np.argsort(f_offspring)
            x_offspring = x_offspring[order]
            
            # Update mean
            x_new = np.sum(w[:, np.newaxis] * x_offspring[:mu], axis=0)
            
            # Update step size
            y_old = (x_new - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * np.linalg.solve(Csqrt, y_old)
            
            # Condition for sigma update
            norm_p_sigma = np.linalg.norm(p_sigma)
            norm_p_sigma_ref = np.sqrt(1 - (1 - c_sigma)**(2 * gen)) * np.linalg.norm(np.linalg.solve(Csqrt, np.ones(D)))
            
            if norm_p_sigma / norm_p_sigma_ref < (1.4 + 2 / (D + 1)):
                sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / norm_p_sigma_ref - 1))
                
            # Update covariance
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * y_old
            
            hsig = 1.0 if norm_p_sigma / norm_p_sigma_ref < (1.4 + 2 / (D + 1)) else 0.0
            
            # Rank-mu update
            y_mu = y[:mu]
            C_update = np.sum(w_i * np.outer(y_i, y_i) for w_i, y_i in zip(w[:mu], y_mu))
            
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - hsig) * c_c * (2 - c_c) * C) + c_mu * C_update
            
            # Symmetry
            C = (C + C.T) / 2.0
            
            # Update square roots
            Csqrt = update_chol(C)
            
            x_mean = x_new
        
        # Use remaining budget for a local refinement if possible
        remaining = budget - calls_made
        if remaining > 0:
            # Simple local search with decreasing step size
            step = sigma * 0.5
            for _ in range(remaining):
                if time.time() > deadline - safety_margin:
                    break
                y = rng.standard_normal(D)
                x_try = np.clip(best_x + step * y, -5, 5)
                f_try = func(x_try)
                if f_try < best_f:
                    best_x = x_try
                    best_f = f_try
                else:
                    step *= 0.9
                    
        return best_x, best_f
