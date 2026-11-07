import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t_start = time.time()
        # Use 280s to be safe against the 300s hard limit
        deadline = t_start + 280.0
        
        def time_left():
            return deadline - time.time()

        # CMA-ES Parameters
        # Population size lambda
        lam = 4 + int(3 * np.log(dim))
        # For dim=5, lam=8
        
        # Initial mean
        m = np.zeros(dim)
        
        # Initial step size
        # A good heuristic for initial sigma is to cover the box or a fraction of it
        # Box is [-5, 5], width 10. Sigma 1.0 is reasonable for 5D.
        sigma = 1.0
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Inverse square root of C (for sampling)
        # We need to sample from N(0, C). x = m + sigma * B * z
        # where B B^T = C. We can use Cholesky or eigen decomposition.
        # For efficiency, we maintain the square root matrix D such that C = D D^T.
        # Initially D = I.
        D = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Learning rates
        # Standard CMA-ES parameters
        mu = lam // 2  # Number of selected individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Damping for sigma update
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        
        # Initial best
        best_x = m.copy()
        best_f = float('inf')
        calls = 0
        
        # Evaluate initial mean to get a starting point
        # This counts as 1 call
        f_m = func(m)
        calls += 1
        best_f = f_m
        best_x = m.copy()
        
        # Main Loop
        while calls + lam <= budget and time_left() > 1.0:
            
            # Generate population
            pop = np.zeros((lam, dim))
            fitness = np.zeros(lam)
            
            for i in range(lam):
                # Sample z ~ N(0, I)
                z = np.random.standard_normal(dim)
                # x = m + sigma * D @ z
                # D is the square root of C
                pop[i] = m + sigma * (D @ z)
                
                # Evaluate
                fitness[i] = func(pop[i])
                calls += 1
                
                if fitness[i] < best_f:
                    best_f = fitness[i]
                    best_x = pop[i].copy()
                    
            # Sort by fitness
            sorted_idx = np.argsort(fitness)
            # Select top mu individuals
            selected_idx = sorted_idx[:mu]
            y_selected = pop[selected_idx]
            
            # Weighted mean of selected
            m_new = np.sum(weights[:, None] * y_selected, axis=0)
            
            # Update evolution path p_sigma
            # We need the inverse of D to transform y - m into whitened space
            # D_inv = inv(D). Since D is upper triangular (from Cholesky) or symmetric?
            # Standard CMA-ES uses Cholesky factor L such that C = L L^T.
            # Let's stick to the standard update rules which assume C is maintained.
            # To keep it simple and robust, let's use the eigen-decomposition approach 
            # or just update C directly and recompute the factorization every step.
            # For dim=5, eigen decomposition is cheap.
            
            # Actually, let's use the standard CMA-ES update which requires C and its inverse/sqrt.
            # To avoid complex matrix inversion bugs, let's use a simpler variant 
            # or ensure the matrix updates are correct.
            
            # Let's restart the matrix management to be safer:
            # We maintain C. We need to sample from C.
            # x = m + sigma * L @ z, where L L^T = C.
            
            # Recompute L from C
            L = np.linalg.cholesky(C)
            
            # Transform y - m to whitened space: w = L^-1 (y - m)
            # Solve L w = (y - m)
            y_minus_m = y_selected - m
            # np.linalg.solve(L, y_minus_m.T).T
            w_selected = np.linalg.solve(L, y_minus_m.T).T
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * np.sum(weights[:, None] * w_selected, axis=0)
            
            # h_sigma condition
            chi_n = np.sqrt(dim) * (1 - 1 / (2 * dim**2))
            h_sigma = 1 if np.sqrt(np.sum(p_sigma**2) / (1 - (1 - c_sigma)**(2 * (calls // lam)))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            
            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * np.sum(weights[:, None] * w_selected, axis=0)
            
            # Update C
            # C = (1 - c_1 - c_mu) * C + c_1 * (p_c p_c^T + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * sum(w_i y_i y_i^T)
            # Note: y_i y_i^T in the formula refers to the whitened vectors w_i w_i^T?
            # Standard formula: C = (1 - c_1 - c_mu)C + c_1 (p_c p_c^T + (1-h_sigma)c_c(2-c_c)C) + c_mu sum(w_i y_i y_i^T)
            # Where y_i are the whitened steps (w_selected).
            
            rank_one_update = np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C
            rank_mu_update = np.sum(weights[:, None, None] * w_selected[:, :, None] * w_selected[:, None, :], axis=0)
            
            C = (1 - c_1 - c_mu) * C + c_1 * rank_one_update + c_mu * rank_mu_update
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2.0
            # Small regularization for numerical stability
            C += 1e-12 * np.eye(dim)
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Update mean
            m = m_new
            
            # Check time again before next iteration
            if time_left() < 2.0:
                break
                
        return best_x, best_f
