import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        lam = 8  # Population size
        mu = lam // 2
        
        # Weights
        w = np.log(lam / 2 + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mueff = 1.0 / np.sum(w**2)
        
        # Step size control parameters
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        cc = (4 + mueff/n) / (n + 4 + 2*mueff/n)
        
        # Initialization
        m = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        inv_sqrt_C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        best_x = m.copy()
        best_f = np.inf
        
        calls = 0
        t0 = time.time()
        
        # Initial evaluation
        fx = func(m)
        calls += 1
        if fx < best_f:
            best_f = fx
            best_x = m.copy()
            
        while calls < budget:
            # Check time
            if time.time() - t0 > 320:
                break
                
            # Sample population
            z = np.random.standard_normal((lam, n))
            x_pop = m + sigma * (C @ z.T).T
            x_pop = np.clip(x_pop, -5, 5)
            
            # Evaluate
            f_pop = np.array([func(x) for x in x_pop])
            calls += lam
            
            # Sort
            idx = np.argsort(f_pop)
            x_sorted = x_pop[idx]
            f_sorted = f_pop[idx]
            
            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = x_sorted[0].copy()
                
            # Update mean
            m_new = np.sum(w[:, None] * x_sorted[:mu], axis=0)
            
            # Update ps
            v = (m_new - m) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * (1 + cs * mueff)) * (inv_sqrt_C @ v)
            
            # Update sigma
            # sigma = sigma * exp(cs * (||ps||^2/n - 1) / (2 * lam))
            sigma = sigma * np.exp(cs * (np.linalg.norm(ps)**2 / n - 1) / (2 * lam))
            
            # Update pc
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * (1 + cc * mueff)) * v
            
            # Update C
            # delta_sigma approximation
            chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
            if np.linalg.norm(ps) > 1.4 * chi_n:
                delta_sigma = 1.0
            else:
                delta_sigma = 0.0
                
            term1 = c1 * (np.outer(pc, pc) + (1 - delta_sigma) * cc * (2 - cc) * np.eye(n))
            
            diffs = (x_sorted[:mu] - m) / sigma
            term2 = cmu * np.sum(w[:, None, None] * diffs[:, :, None] * diffs[:, None, :], axis=0)
            
            C = (1 - c1 - cmu) * C + term1 + term2
            
            # Update inv_sqrt_C
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            inv_sqrt_C = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
            
            m = m_new
            m = np.clip(m, -5, 5)
            
        return best_x, best_f
