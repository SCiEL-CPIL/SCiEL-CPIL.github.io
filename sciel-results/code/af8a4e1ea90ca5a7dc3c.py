import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lam = 10
        mu = lam // 2
        
        # Weights
        w = np.log(np.arange(1, mu + 1) + 0.5) - np.log(mu + 0.5)
        w = w / np.sum(w)
        mu_eff = np.sum(w) ** 2 / np.sum(w ** 2)
        
        # Strategy parameters (fixed c_c)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = 3 / (dim + 4)  # Fixed: was using mu_eff/mu_eff which is 1
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        c_1_mu = c_1 + c_mu
        
        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0
        
        # Multi-start: try a few initial locations, then run CMA-ES
        # with stall-based restarts
        max_restarts = 3
        stall_threshold_calls = 80  # No improvement in this many calls triggers restart
        
        for restart_iter in range(max_restarts):
            if calls >= budget or time.time() >= deadline:
                break
            
            # Initialize CMA-ES state
            if restart_iter == 0:
                m = np.zeros(dim)
            else:
                # Restart: random location near best or fully random
                if restart_iter == 1:
                    m = np.clip(best_x + 2.0 * np.random.standard_normal(dim), -5, 5)
                else:
                    m = np.random.uniform(-4, 4, dim)
            
            sigma = 2.0
            C = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            
            # Evaluate initial mean
            f_m = func(m)
            calls += 1
            if f_m < best_f:
                best_f = f_m
                best_x = m.copy()
            
            last_improvement_call = calls
            no_improvement_count = 0
            
            while calls < budget:
                if time.time() >= deadline:
                    break
                
                # 1. Sample population
                Z = np.random.standard_normal((lam, dim))
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
                
                Y = Z @ C_sqrt.T
                X = m + sigma * Y
                X = np.clip(X, -5, 5)
                
                # Evaluate
                F = np.array([func(x) for x in X])
                calls += lam
                
                # Update best
                idx_min = np.argmin(F)
                if F[idx_min] < best_f:
                    best_f = F[idx_min]
                    best_x = X[idx_min].copy()
                    no_improvement_count = 0
                else:
                    no_improvement_count += lam
                
                # Check for stall
                if no_improvement_count >= stall_threshold_calls:
                    break  # Trigger restart
                
                # 2. Sort
                idx_sort = np.argsort(F)
                X_sorted = X[idx_sort]
                F_sorted = F[idx_sort]
                
                # 3. Update Mean
                m_new = np.sum(w[:, np.newaxis] * X_sorted[:mu], axis=0)
                
                # 4. Update Step Size (CSA)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                delta = (m_new - m) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(c_1_mu * (2 - c_1_mu)) * (C_inv_sqrt @ delta)
                
                norm_p_sigma = np.linalg.norm(p_sigma)
                chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (25 * dim ** 2))
                
                sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
                
                # 5. Update Covariance
                p_c = (1 - c_c) * p_c + c_c * delta
                Y_sorted = (X_sorted[:mu] - m) / sigma
                C_new = (1 - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c)
                for i in range(mu):
                    C_new += c_mu * w[i] * np.outer(Y_sorted[i], Y_sorted[i])
                
                C_new = (C_new + C_new.T) / 2
                m = m_new
                C = C_new
        
        return best_x, best_f
