import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters for dim=5, budget=500
        # lambda = 4 + 3*ln(5) ~ 9 -> use 10 for even split? Or standard.
        # Standard lambda for CMA-ES: 4 + 3*ln(N)
        lam = max(4, int(4 + 3 * np.log(dim))) 
        if lam % 2 == 0:
            lam += 1 # Make odd? Not strictly necessary but often preferred. Let's stick to standard.
        # Actually, let's just use lam = 10 for dim=5 as it's a common choice and 500/10=50 gens is good.
        lam = 10 
        
        mu = lam // 2
        # Weights: log-weights
        weights = np.log(lam + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1.0 - c1, 2 * (mueff - 2 + 1.0/mueff) / ((dim + 2)**2 + mueff))
        ds = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs
        chiN = np.sqrt(dim) * (1.0 - 1.0/(4.0*dim) + 1.0/(21.0*dim**2))
        
        # Initialization
        # Start at origin. The problem has a hidden shift, but 0 is a safe neutral start.
        # Initial sigma: The box is [-5,5]. A step of 2.0 allows covering the space in ~5 steps.
        x = np.zeros(dim)
        sigma = 2.0
        
        C = np.eye(dim)
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        calls = 0
        best_x = x.copy()
        best_f = func(best_x)
        calls += 1
        
        start_time = time.time()
        deadline = start_time + 280.0
        
        # Precompute Cholesky factor for initial sampling
        try:
            L = np.linalg.cholesky(C)
        except:
            L = np.eye(dim)
            
        while calls + lam <= budget:
            if time.time() > deadline:
                break
                
            # Generate offspring
            ys = []
            fs = []
            for i in range(lam):
                z = np.random.randn(dim)
                y = x + sigma * (L @ z)
                y = np.clip(y, -5.0, 5.0)
                ys.append(y)
                fy = func(y)
                fs.append(fy)
                calls += 1
                
            # Check if we ran out of budget mid-generation (shouldn't happen due to while condition, but safe)
            if calls >= budget:
                break
                
            # Sort
            idx = np.argsort(fs)
            ys = [ys[i] for i in idx]
            fs = [fs[i] for i in idx]
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = ys[0].copy()
                
            # Update mean
            x_old = x.copy()
            ymean = np.zeros(dim)
            for i in range(mu):
                ymean += weights[i] * ys[i]
            x = x_old + ymean
            
            # Update paths
            # C_sqrt_inv
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # pc
            pc = (1.0 - cs) * pc + cs * np.sqrt(mueff) * C_sqrt_inv @ (ymean / sigma)
            
            # ps
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) * C_sqrt_inv @ (ymean / sigma)
            
            # Sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chiN - 1.0))
            
            # C update
            # Rank-one
            C = (1.0 - c1 - cmu) * C
            C += c1 * (np.outer(pc, pc) + (1.0 - (1.0 - cs)**2) * mueff * np.eye(dim))
            
            # Rank-mu
            # Note: ys are absolute points. The update requires (ys[i] - x_old)
            for i in range(mu):
                diff = ys[i] - x_old
                C += cmu * weights[i] * np.outer(diff, diff)
                
            # Symmetrize
            C = (C + C.T) / 2.0
            
            # Update Cholesky for next iteration
            try:
                L = np.linalg.cholesky(C)
            except:
                # If C is not SPD, regularize
                C += 1e-6 * np.eye(dim)
                try:
                    L = np.linalg.cholesky(C)
                except:
                    L = np.eye(dim)
                    
            # Safety bounds for sigma
            if sigma > 10.0:
                sigma = 10.0
            if sigma < 1e-6:
                sigma = 1e-6

        # If we have remaining budget and time, do a few single-point evaluations?
        # Not necessary for CMA-ES, but if we broke early due to time, we just return.
        # If we broke because calls + lam > budget, we might have 1-4 calls left.
        # We can use them to evaluate random perturbations of the best point?
        # Or just stop. CMA-ES is robust enough.
        
        return best_x, best_f
