import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lam = 10
        # Number of best individuals
        mu = lam // 2
        
        # Weights
        w = np.log(np.arange(1, mu + 1) + 0.5) - np.log(mu + 0.5)
        w = w / np.sum(w)
        mu_eff = np.sum(w) ** 2 / np.sum(w ** 2)
        
        # Strategy parameters
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 - mu_eff / mu_eff) / (dim + 4) # Simplified, usually close to 3/(n+4)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Initialization
        # Start at 0 or random? Random is safer for shifted functions.
        # But CMA-ES is robust. Let's start at 0.
        m = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        
        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Pre-compute C^(-1/2) for CSA? 
        # We need to update C^(-1/2) every step? That's expensive.
        # For n=5, it's fine.
        
        best_x = m.copy()
        best_f = np.inf
        calls = 0
        
        # Initial evaluation? 
        # CMA-ES samples population.
        
        while calls < budget:
            if time.time() >= deadline:
                break
            
            # 1. Sample population
            # Generate standard normal samples
            Z = np.random.standard_normal((lam, dim))
            # Transform to C
            # We need C^1/2. We can use eigendecomposition.
            # For n=5, this is fast.
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip eigvals to avoid negative
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            Y = Z @ C_sqrt.T # (lam, dim)
            X = m + sigma * Y
            
            # Clip to bounds
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            calls += lam
            
            # Update best
            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = X[idx_min].copy()
            
            # 2. Sort
            idx_sort = np.argsort(F)
            X_sorted = X[idx_sort]
            F_sorted = F[idx_sort]
            
            # 3. Update Mean
            m_new = np.sum(w[:, np.newaxis] * X_sorted[:mu], axis=0)
            
            # 4. Update Step Size (CSA)
            # p_sigma = (1 - c_sigma) p_sigma + c_sigma * sqrt(c_(1+mu) (2 - c_(1+mu))) * C^(-1/2) (m_new - m) / sigma
            # c_(1+mu) = c_1 + c_mu
            c_1_mu = c_1 + c_mu
            # We need C^(-1/2). 
            # C_inv_sqrt = eigvecs @ diag(1/sqrt(eigvals)) @ eigvecs.T
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            delta = (m_new - m) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(c_1_mu * (2 - c_1_mu)) * (C_inv_sqrt @ delta)
            
            # Norm check
            norm_p_sigma = np.linalg.norm(p_sigma)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (25 * dim ** 2))
            
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            
            # 5. Update Covariance
            # p_c = (1 - c_c) p_c + c_c (m_new - m) / sigma
            p_c = (1 - c_c) * p_c + c_c * delta
            
            # C = (1 - c_1 - c_mu) C + c_1 p_c p_c^T + c_mu sum(w_i y_i y_i^T)
            # y_i = (x_i - m) / sigma
            Y_sorted = (X_sorted[:mu] - m) / sigma
            C_new = (1 - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c)
            for i in range(mu):
                C_new += c_mu * w[i] * np.outer(Y_sorted[i], Y_sorted[i])
            
            # Symmetrize
            C_new = (C_new + C_new.T) / 2
            
            m = m_new
            C = C_new
            
        return best_x, best_f
