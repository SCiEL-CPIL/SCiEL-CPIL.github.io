import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = time.time()
        self.deadline = 300.0

    def __call__(self, func):
        d = self.dim
        N = 10
        mu = N // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))
        
        # Initialization
        m = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)
        B = np.eye(d)
        D = 1.0
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        rng = np.random.default_rng(0)
        
        calls = 0
        best_x = m.copy()
        best_f = np.inf
        
        chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))
        
        for gen in range(self.budget // N + 1):
            if time.time() > self.deadline:
                break
            
            # Generate population
            pop = np.zeros((N, d))
            for i in range(N):
                if calls >= self.budget:
                    break
                # Sample
                z = B @ (D * rng.standard_normal(d))
                pop[i] = m + sigma * z
                calls += 1
            
            if calls >= self.budget:
                break
            
            # Evaluate
            fvals = np.array([func(np.clip(pop[i], -5, 5)) for i in range(N)])
            
            # Track best
            idx_min = np.argmin(fvals)
            if fvals[idx_min] < best_f:
                best_f = fvals[idx_min]
                best_x = pop[idx_min].copy()
                
            # Sort
            idx = np.argsort(fvals)
            pop_sorted = pop[idx]
            fvals_sorted = fvals[idx]
            
            # Update mean
            m_new = np.sum(w[:, np.newaxis] * pop_sorted[:mu], axis=0)
            
            # Update ps
            # z_shift = B.T @ (m_new - m) / sigma
            z_shift = B.T @ ((m_new - m) / sigma)
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_shift
            
            # Update pc
            # hc = 1 if ||ps||^2 < (1 - (1 - c_sigma)^(2 * gen)) * (4 + c_sigma) * mu_eff else 0
            # Simplified hc
            hc = 1.0
            pc = (1 - c_c) * pc + hc * np.sqrt(c_c * (2 - c_c) * mu_eff) * z_shift
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma * (np.linalg.norm(ps) / np.sqrt(mu_eff) - chi_n))
            sigma = np.clip(sigma, 1e-8, 10.0)
            
            # Update C
            # Rank-one
            v1 = pc
            c1_term = (1 + c_1 * hc * (hc - 1 + c_c) - c_1)
            C = c1_term * C + c_1 * np.outer(v1, v1)
            
            # Rank-mu
            Y = (pop_sorted[:mu] - m) / sigma
            C_mu = Y.T @ (w[:mu, np.newaxis] * Y)
            C += c_mu * C_mu
            
            C = (C + C.T) / 2
            
            # Update B, D
            try:
                L = np.linalg.cholesky(C)
                D = 1.0 / np.sqrt(np.diag(L))
                B = np.linalg.inv(L)
            except:
                C += 1e-6 * np.eye(d)
                L = np.linalg.cholesky(C)
                D = 1.0 / np.sqrt(np.diag(L))
                B = np.linalg.inv(L)
                
            m = m_new

        return best_x, best_f
