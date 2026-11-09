import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 300.0
        safety_margin = 5.0
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lambda_ = 10  # Population size
        mu = lambda_ // 2  # Number of parents
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Initial state
        x = np.zeros(dim)
        sigma = 2.0 # Initial step size
        C = np.eye(dim) # Covariance matrix
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = x.copy()
        best_f = func(best_x)
        
        calls_used = 1
        gen = 0
        
        # Pre-compute constants
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        while calls_used + lambda_ <= budget:
            if time.time() - start_time > time_limit - safety_margin:
                break
            
            gen += 1
            
            # Compute Cholesky factor of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, add small identity
                C += 1e-8 * np.eye(dim)
                L = np.linalg.cholesky(C)
            
            # 1. Sampling
            Y = []
            for i in range(lambda_):
                z = np.random.normal(0, 1, dim)
                y = x + sigma * (L @ z)
                Y.append(y)
            Y = np.array(Y)
            
            # 2. Evaluation
            F = np.array([func(y) for y in Y])
            calls_used += lambda_
            
            # Update best
            idx_best = np.argmin(F)
            if F[idx_best] < best_f:
                best_f = F[idx_best]
                best_x = Y[idx_best].copy()
                
            # 3. Update CMA-ES State
            # Sort indices by fitness
            idx_sorted = np.argsort(F)
            # Take top mu
            Y_parents = Y[idx_sorted[:mu]]
            
            # Update mean
            x_old = x.copy()
            x_new = np.average(Y_parents, axis=0, weights=weights)
            
            # Update step size
            delta = x_new - x_old
            # Solve L @ v = delta
            v = np.linalg.solve(L, delta)
            delta_white = v / sigma
            
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * delta_white
            
            # Heaviside function for sigma update
            norm_p_sigma = np.linalg.norm(p_sigma)
            expected_norm = np.sqrt(1 - (1 - c_sigma)**(2 * gen))
            threshold = (1.4 + 2 / (dim + 1)) * chi_n
            h_sigma = 1 if norm_p_sigma / expected_norm < threshold else 0
            
            sigma = sigma * np.exp(c_sigma * (norm_p_sigma / chi_n - 1))
            
            # Update covariance matrix C
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (delta / sigma)
            
            # Rank-one update
            c_1_term = c_1 * (np.outer(delta, delta) / sigma**2 - (1 - h_sigma) * c_c * (2 - c_c) * np.outer(p_c, p_c))
            
            # Rank-mu update
            Z_parents = (Y_parents - x_old) / sigma
            # Solve L @ Z_white.T = Z_parents.T
            Z_white_T = np.linalg.solve(L, Z_parents.T)
            Z_white = Z_white_T.T
            
            rank_mu_mat = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu_mat += weights[i] * np.outer(Z_white[i], Z_white[i])
            
            c_mu_term = c_mu * (rank_mu_mat - C)
            
            C = C + c_1_term + c_mu_term
            
            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) <= 1e-12:
                C += (1e-12 - np.min(eigvals)) * np.eye(dim)
            
            x = x_new

        return best_x, best_f
