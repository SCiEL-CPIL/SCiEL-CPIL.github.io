import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        rng = np.random.default_rng(0)
        
        n = self.dim
        lam = 4 + int(3 * np.log(n))
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mu_eff = 1 / np.sum(w**2)
        
        sigma = 2.0
        m = rng.uniform(-5, 5, n)
        C = np.eye(n)
        C_inv_sqrt = np.eye(n)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        c_c = (4 - mu_eff + 0.3 * (mu_eff - 1)) / ((n + 2) ** 2 + mu_eff)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        best_x = m.copy()
        best_f = func(best_x)
        calls_made = 1
        
        max_iters = (self.budget - 1) // lam
        
        for _ in range(max_iters):
            if time.time() >= deadline:
                break
            
            m_old = m.copy()
            
            # Generate
            Y = np.array([np.clip(m + sigma * (C_inv_sqrt @ rng.standard_normal(n)), -5, 5) for _ in range(lam)])
            
            # Evaluate
            F = np.array([func(y) for y in Y])
            calls_made += lam
            
            # Sort
            idx = np.argsort(F)
            Y = Y[idx]
            F = F[idx]
            
            if F[0] < best_f:
                best_f = F[0]
                best_x = Y[0].copy()
            
            # Update Mean
            m = np.sum(w[:, np.newaxis] * Y[:mu], axis=0)
            
            # Update Step Size
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (m - m_old) / sigma)
            
            # Heuristic to avoid sigma explosion
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls_made // lam + 1))) < (1.4 + 2 / (n + 1)) * np.sqrt(mu_eff + 1):
                sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                # Standard update: sigma = sigma * exp(c_sigma * (||p_sigma||/sqrt(mu_eff) - 1))
                # Wait, the standard formula is:
                # sigma = sigma * exp( c_sigma / d_sigma * ( ||p_sigma|| / sqrt(mu_eff) - chi_n ) )
                # d_sigma = 1 + 2/max(0, (mu_eff-1)/(n+1)) + c_sigma
                # chi_n = sqrt(n) * (1 - 1/(4n) + 1/(21n^2))
                
                # Let's use the standard update properly.
                d_sigma = 1 + 2.0 / max(0.0, (mu_eff - 1.0) / (n + 1.0)) + c_sigma
                chi_n = np.sqrt(n) * (1 - 1.0 / (4.0 * n) + 1.0 / (21.0 * n**2))
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - chi_n))
            
            # Update Covariance
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (m - m_old) / sigma
            
            # Rank-1 and Rank-mu update
            C = (1 + c_1 * (c_mu + nu) - c_1) * C + c_1 * np.outer(p_c, p_c)
            C += c_mu * np.sum(w_i * np.outer(Y_i - m_old, Y_i - m_old) / sigma**2 for w_i, Y_i in zip(w, Y[:mu]))
            
            # Ensure positive definite
            # Symmetrize
            C = (C + C.T) / 2.0
            
            # Recompute C_inv_sqrt
            # Use eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C_inv_sqrt = (eigvecs / np.sqrt(eigvals)) @ eigvecs.T
            
            # Optional: Check convergence
            if np.linalg.norm(m - m_old) < 1e-8 * sigma:
                break
                
        return best_x, best_f
