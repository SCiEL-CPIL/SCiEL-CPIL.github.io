import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        N = self.budget
        start_time = time.time()
        
        # CMA-ES Parameters
        # Choose mu based on N and d
        # For N=500, d=5, mu=5 allows 100 generations.
        mu = max(1, min(5, N // 2)) 
        # Actually, let's use a slightly larger mu to get better gradient estimation if possible, 
        # but small mu is faster per generation.
        # Let's use mu = max(1, int(np.sqrt(N)))? No.
        # Standard is mu = floor(N/2). But that's 250. 2 generations. Too few.
        # So we MUST use small mu.
        mu = max(1, d) # mu=5
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1 / np.sum(w**2)
        
        # Strategy parameters
        c_sigma = (mueff + 2) / (d + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        c_1 = 2 / ((d + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2)**2 + mueff))
        
        # Initialization
        x_mean = np.zeros(d)
        sigma = 2.0  # Start with a reasonably large step size to explore [-5,5]
        C = np.eye(d)
        C_inv_sqrt = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        chi_d = np.sqrt(d) * (1 - 1/(4*d) + 1/(21*d**2))
        
        calls = 0
        best_x = x_mean.copy()
        best_f = np.inf
        
        def eval_func(x):
            nonlocal calls
            if calls >= N:
                return np.inf
            if time.time() - start_time > 290:
                return np.inf
            calls += 1
            try:
                return float(func(x))
            except Exception:
                return np.inf

        # Initial evaluation
        f_mean = eval_func(x_mean)
        if f_mean < best_f:
            best_f = f_mean
            best_x = x_mean.copy()
            
        while calls < N:
            if time.time() - start_time > 290:
                break
                
            # Generate population
            Z = np.random.randn(mu, d)
            Y = C_inv_sqrt @ Z.T
            P = x_mean + sigma * Y.T
            P = np.clip(P, -5, 5)
            
            F = np.array([eval_func(p) for p in P])
            
            # Update best
            for i in range(mu):
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = P[i].copy()
                    
            # Sort
            idx = np.argsort(F)
            P = P[idx]
            F = F[idx]
            
            # Update mean
            x_mean_old = x_mean.copy()
            x_mean = x_mean + sigma * np.sum(w[:, None] * P[:mu], axis=0)
            x_mean = np.clip(x_mean, -5, 5)
            
            # Update step size
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (x_mean - x_mean_old)
            
            # Update covariance matrix
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) / sigma * (x_mean - x_mean_old)
            
            h = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * calls / N)) < (1.4 + 2 / (d + 1)) * chi_d)
            delta_h = 1 if h else 0
            
            C = (1 - c_1 - c_mu + mu * delta_h * (1 - c_c) * (1 - c_c)) * C \
                + c_1 * (np.outer(pc, pc) + (1 - delta_h) * (1 - c_c**2) * C) \
                + c_mu * np.sum(w[:, None, None] * np.outer(P[:mu].T - x_mean_old, P[:mu].T - x_mean_old) / sigma**2, axis=2)
            
            C = (C + C.T) / 2
            try:
                U, s, Vt = np.linalg.svd(C)
                C_inv_sqrt = Vt.T @ np.diag(1.0 / np.sqrt(s)) @ U.T
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(d)
                
            if sigma < 1e-12:
                break

        return best_x, best_f
