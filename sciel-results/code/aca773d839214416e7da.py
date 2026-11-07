import numpy as np
import time

class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        
        # CMA-ES parameters
        lam = 4 + int(3 * np.log(dim))
        if lam < 4: lam = 4
        
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w**2)
        
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
        cc = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
        
        m = np.random.uniform(-5, 5, dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chiN = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        best_x = m.copy()
        best_f = func(m)
        calls = 1
        gen = 0
        
        while calls + lam <= budget:
            if time.time() - start_time > 280:
                break
            
            # Sample population
            # Use Cholesky for efficiency and correctness
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset
                C = np.eye(dim)
                L = np.eye(dim)
                sigma = 1.0
                
            Z = np.random.randn(lam, dim)
            X = m[None, :] + sigma * (Z @ L.T)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            calls += lam
            
            # Sort
            idx = np.argsort(F)
            X = X[idx]
            F = F[idx]
            
            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = X[0].copy()
                
            # Update mean
            w_mu = w[:mu]
            m_new = np.sum(w_mu[:, None] * X[:mu], axis=0)
            
            # Update evolution paths
            pc = (1 - cc) * pc + cc / np.sqrt(cc * (2 - cc) * mueff) * (C @ (m_new - m)) / sigma
            hs = (np.linalg.norm(pc) / (np.sqrt(1 - (1 - cc)**(gen + 1)) * chiN)) < (1.4 + 2 / (dim + 1))
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (m_new - m) / sigma
            
            # Update sigma
            sigma *= np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1))
            
            # Update covariance
            C = (1 + c1 * (hs * (1 + (1 - hs) * cc / (1 - cc)) - 1) + cmu) * C
            diff = X[:mu] - m
            C += c1 * np.outer(pc, pc)
            C += cmu * np.sum(w_mu[:, None, None] * diff[:, :, None] * diff[:, None, :], axis=0)
            
            m = m_new
            gen += 1
            
            if sigma < 1e-10:
                break
                
        return best_x, best_f
