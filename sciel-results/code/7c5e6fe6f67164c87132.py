"""CMA-ES based optimizer for BBOB f3 (Rastrigin) with hidden transformations."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with large population for multimodal landscape exploration."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        # CMA-ES parameters
        n = self.dim
        budget = self.budget
        
        # Initial population size: lambda = 4 + floor(3 * log(n + 2))
        # For n=5: lambda = 4 + floor(3*2.9) = 4 + 8 = 12, but we want larger for multimodal
        # Let's use lambda = 30 as suggested
        lam = 30
        mu = lam // 2  # number of parents
        
        # Weights
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w**2)
        
        # Step size
        sigma = 2.0
        
        # Initial mean: start at origin (0,0,0,0,0) which is often good for Rastrigin
        m = np.zeros(n)
        
        # Covariance matrix: identity
        C = np.eye(n)
        
        # Precompute square root of C (initially identity)
        C_sqrt = np.eye(n)
        
        # Evolution paths
        ps = np.zeros(n)
        pc = np.zeros(n)
        
        # CMAs
        # c_sigma = (mueff + 2) / (n + mueff + 5)
        c_sigma = (mueff + 2) / (n + mueff + 5)
        # d_sigma = 1 + 2*max(0, sqrt((mueff-1)/(n+1)) - 1) + c_sigma
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        # c_c = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        # c_1 = 2 / ((n + 1.3)^2 + mueff)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        # c_mu = min(1 - c_1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)^2 + mueff))
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        # Count evaluations
        evals = 0
        best_x = None
        best_f = np.inf
        
        # Initial evaluation
        x0 = np.clip(m, -5, 5)
        f0 = func(x0)
        evals += 1
        best_x = x0.copy()
        best_f = f0
        
        # Main loop
        while evals < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate population
            X = np.zeros((lam, n))
            F = np.zeros(lam)
            
            for i in range(lam):
                # Check time periodically
                if i % 5 == 0 and time.time() > deadline:
                    break
                
                z = rng.standard_normal(n)
                # Transform z using C_sqrt
                # C_sqrt is the square root of C, so x = m + sigma * C_sqrt @ z
                x = m + sigma * C_sqrt @ z
                x = np.clip(x, -5, 5)
                X[i] = x
                F[i] = func(x)
                evals += 1
                
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = x.copy()
            
            if evals >= budget:
                break
            
            # Sort by fitness
            idx = np.argsort(F)
            X = X[idx]
            F = F[idx]
            
            # Update mean
            m_old = m.copy()
            m = np.sum(w[:, None] * X[:mu], axis=0)
            
            # Update evolution paths
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * C_sqrt @ (m - m_old)
            if sigma > 0:
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * C_sqrt @ (m - m_old)
            
            # Heuristic h_sigma
            h_sigma = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (evals - n))) / np.sqrt(1 - (1 - c_sigma)**(2 * (evals - n))) if evals > n else 1.0
            h_sigma = min(1.0, h_sigma / (1 - (1 - c_sigma)**(evals / n)))
            
            # Update pc
            pc = (1 - c_c) * pc + h_sigma * c_c * (m - m_old) / sigma
            
            # Update C
            # Rank-one update
            delta_m = (m - m_old) / sigma
            C = (1 + c_1 * (h_sigma - 1) + c_mu) * C + c_1 * (h_sigma - 1) * np.outer(pc, pc) + c_mu * np.sum(w[:, None, None] * np.outer(X[:mu] - m_old, X[:mu] - m_old) / sigma**2, axis=0)
            
            # Ensure symmetry and positive semi-definite
            C = (C + C.T) / 2
            
            # Update sigma
            # chi_n = sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Re-estimate C_sqrt
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            except:
                C_sqrt = np.eye(n)
            
            # Clip sigma to reasonable range
            sigma = np.clip(sigma, 1e-10, 10.0)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
