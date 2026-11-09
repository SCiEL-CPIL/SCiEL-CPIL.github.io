import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = time.time() + 300  # 300 seconds safety margin

    def __call__(self, func):
        N = self.dim
        # Population size lambda
        lam = 4 + int(3 * np.log(N))
        # Number of selected individuals
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w**2)
        
        # Parameters
        c_sigma = (mueff + 2) / (N + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (N + 1)) - 1) + c_sigma
        c_c = (4 - mueff + c_sigma) / (N + 4 - mueff)
        c_1 = 2 / ((N + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1/mueff) / ((N + 2)**2 + mueff))
        chi_N = np.sqrt(N) * (1 - 1/(4*N) + 1/(32*N**2))
        
        # Initial state
        x = np.zeros(N)
        sigma = 1.0
        p_sigma = np.zeros(N)
        p_c = np.zeros(N)
        C = np.eye(N)
        # Cholesky factor of C
        C_12 = np.eye(N)
        
        best_x = x
        best_f = float('inf')
        
        calls_made = 0
        
        while calls_made < self.budget:
            # Check time
            if time.time() > self.deadline:
                break
                
            # Generate population
            Z = np.random.randn(lam, N)
            Y = x + sigma * (Z @ C_12.T)
            
            # Clip to bounds
            Y = np.clip(Y, -5, 5)
            
            # Evaluate
            F = np.array([func(y) for y in Y])
            calls_made += lam
            
            # Sort
            idx = np.argsort(F)
            Y_sorted = Y[idx]
            F_sorted = F[idx]
            
            # Update best
            if F_sorted[0] < best_f:
                best_f = F_sorted[0]
                best_x = Y_sorted[0].copy()
                
            # Selection
            Y_sel = Y_sorted[:mu]
            
            # New mean
            x_new = np.sum(w[:, None] * Y_sel, axis=0)
            
            # Update step size control variable
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (C_12.T @ (x_new - x))
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_N - 1))
            
            # Update evolution path for covariance
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mueff) * (x_new - x) / sigma
            
            # Update covariance matrix
            delta_h = (1 + np.sqrt(2 * c_c - c_c**2)) * (np.linalg.norm(p_c)**2 / N - 1)
            rank_one = np.outer(p_c, p_c)
            rank_mu = np.sum(w[:, None, None] * np.outer(Y_sel - x, Y_sel - x), axis=0)
            
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * rank_one + c_mu * rank_mu
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            try:
                C_12 = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If Cholesky fails, regularize C
                C += 1e-6 * np.eye(N)
                try:
                    C_12 = np.linalg.cholesky(C)
                except:
                    C_12 = np.eye(N)
                    
            x = x_new
            
        return best_x, best_f
