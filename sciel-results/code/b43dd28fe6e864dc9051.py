import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0

        eval_count = 0
        best_x = np.zeros(d)
        best_f = float('inf')

        def eval_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget or time.time() > deadline:
                return float('inf')
            x = np.clip(x, -5, 5)
            f = func(x)
            eval_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # CMA-ES Parameters
        lam = 9
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w**2)
        
        m = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)
        ps = np.zeros(d)
        
        cc = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        cs = (mu_eff + 2) / (d + mu_eff + 5)
        c1 = 2 / ((d + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        # Precompute
        B = np.linalg.cholesky(C)
        D = np.sqrt(np.diag(C))
        
        gen = 0
        while eval_count < budget:
            xs = np.zeros((lam, d))
            fs = np.zeros(lam)
            
            for i in range(lam):
                if eval_count >= budget: break
                z = np.random.randn(d)
                xi = m + sigma * (B @ z)
                xs[i] = xi
                fs[i] = eval_func(xi)
                
            if eval_count >= budget: break
            
            idx = np.argsort(fs)
            xs = xs[idx]
            fs = fs[idx]
            
            m_old = m.copy()
            m = np.sum(w[:, None] * xs[:mu], axis=0)
            
            # Update ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (B @ (m - m_old))
            
            # Update sigma
            # hsig check
            # norm(ps) / sqrt(1 - (1-cs)^(2*(gen+1)))
            # We approximate the threshold
            if np.linalg.norm(ps) / D < (1 + 1.5/d) * np.sqrt(d + 1 - 1/d):
                 sigma *= np.exp(cs / (d + 3) * (np.linalg.norm(ps) / D - np.sqrt(d)))
                 
            # Update C
            # Rank-1 update
            y = (m - m_old) / sigma
            C = (1 - c1 - cmu) * C + c1 * (np.outer(ps, ps) + (1 - (1-cs)**2) * C) / sigma**2 + cmu * np.sum(w[:, None, None] * np.outer(xs[:mu] - m, xs[:mu] - m), axis=0) / sigma**2
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update B and D
            try:
                B = np.linalg.cholesky(C)
                D = np.sqrt(np.diag(C))
            except:
                # If Cholesky fails, reset C to identity or diagonal
                C = np.eye(d)
                B = np.eye(d)
                D = np.ones(d)
                
            gen += 1
            
            # Check convergence
            if sigma < 1e-8:
                break

        return best_x, best_f
