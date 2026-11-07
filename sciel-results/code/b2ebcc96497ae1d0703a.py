import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        D = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lam = 4 + int(3 * np.log(D))
        if lam < 2:
            lam = 2
            
        # Number of parents
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w**2)
        
        # Learning rates
        cs = (mueff + 1.7) / (D + mueff + 5)
        c1 = 2 / ((D + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((D + 2)**2 + mueff))
        cc = (4 + mueff/D) / (D + 4 + 2*mueff/D)
        
        # State
        m = np.zeros(D)
        sigma = 1.0
        C = np.eye(D)
        B = np.eye(D)
        Dinv = np.ones(D)
        ps = np.zeros(D)
        pc = np.zeros(D)
        
        calls = 0
        best_x = m.copy()
        best_f = float('inf')
        
        def eval_func(x):
            nonlocal calls
            calls += 1
            return func(x)
            
        # Initial evaluation
        best_f = eval_func(m)
        best_x = m.copy()
        
        while calls < budget:
            # Generate population
            X = np.zeros((lam, D))
            
            # Compute Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Regularize and retry
                C = C + 1e-12 * np.eye(D)
                try:
                    L = np.linalg.cholesky(C)
                except:
                    L = np.eye(D)
                    
            for i in range(lam):
                z = np.random.randn(D)
                x = m + sigma * (L @ z)
                # Clip to box
                x = np.clip(x, -5, 5)
                X[i] = x
                
            # Evaluate population
            F = np.array([eval_func(x) for x in X])
            
            # Update best
            min_idx = np.argmin(F)
            if F[min_idx] < best_f:
                best_f = F[min_idx]
                best_x = X[min_idx].copy()
                
            # Sort by fitness
            order = np.argsort(F)
            Xs = X[order]
            
            # Update mean
            m_old = m.copy()
            m = np.sum(w[:, None] * Xs[:mu], axis=0)
            
            # Update path ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B @ (m - m_old)) / sigma
            
            # Heaviside operator
            norm_ps = np.linalg.norm(ps)
            threshold = (1.4 + 2 / (D + 1)) * np.sqrt((D + 1) / (D - 1))
            hsig = 1.0 if (norm_ps / np.sqrt(1 - (1 - cs)**2)) < threshold else 0.0
            
            # Update path pc
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (B @ (m - m_old)) / sigma
            
            # Update covariance matrix
            delta_mu = (1 + 2 * max(0, (mueff - 1) / (D + 1)) - cc) * cmu
            C = (1 - (c1 + delta_mu)) * C + c1 * (np.outer(ps, ps) + (1 - hsig) * cc * (2 - cc) * np.outer(pc, pc))
            
            for i in range(mu):
                C += w[i] * delta_mu * np.outer(Xs[i] - m_old, Xs[i] - m_old)
                
            # Symmetrize
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp(cs / 5 * (norm_ps / np.sqrt(D) - 1))
            
            # Update B and Dinv (eigen decomposition)
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                B = eigvecs
                Dinv = 1.0 / np.sqrt(eigvals)
            except:
                C = np.eye(D)
                B = np.eye(D)
                Dinv = np.ones(D)
                
            # Safety check for NaNs
            if not np.isfinite(best_f):
                break
                
        return best_x, best_f
