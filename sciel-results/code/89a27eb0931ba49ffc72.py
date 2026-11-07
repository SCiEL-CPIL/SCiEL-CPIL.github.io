import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # lambda = 4 + floor(3 * log(D))
        lam = max(4, int(4 + 3 * np.log(dim)))
        # For D=5, lam = 4 + 3*1.6 = 4+4 = 8? No, log(5)=0.69, 3*0.69=2.07, floor=2. lam=4+6=10.
        
        # We have 500 calls.
        # If lam=10, we have 50 generations.
        
        # Initialization
        # Start at 0, as it's the center of the box and a likely good starting point for shifted Rosenbrock.
        x = np.zeros(dim)
        sigma = 2.0
        
        # Covariance matrix
        C = np.eye(dim)
        # Paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Weights
        mu = lam // 2
        # Weights for top mu individuals
        w = np.log(lam + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mueff = 1 / np.sum(w**2)
        
        # Learning rates
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2)**2 + mueff))
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        calls = 0
        best_x = x.copy()
        best_f = func(best_x)
        calls += 1
        
        start_time = time.time()
        deadline = start_time + 280.0
        
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Generate offspring
            x_old = x.copy()
            ys = []
            # Precompute Cholesky if possible, but C changes every step.
            # For dim=5, Cholesky is very fast.
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback to identity if C is not PD
                L = np.eye(dim)
                
            for i in range(lam):
                z = np.random.randn(dim)
                y = x + sigma * (L @ z)
                y = np.clip(y, -5.0, 5.0)
                ys.append(y)
            
            # Evaluate
            fs = np.array([func(y) for y in ys])
            calls += lam
            
            if calls >= budget:
                idx = np.argmin(fs)
                if fs[idx] < best_f:
                    best_f = fs[idx]
                    best_x = ys[idx].copy()
                break

            # Sort
            sorted_idx = np.argsort(fs)
            ys_sorted = [ys[i] for i in sorted_idx]
            fs_sorted = fs[sorted_idx]
            
            # Update best
            if fs_sorted[0] < best_f:
                best_f = fs_sorted[0]
                best_x = ys_sorted[0].copy()
                
            # Update mean
            ymean = np.sum([w[i] * ys_sorted[i] for i in range(mu)], axis=0)
            x = x + ymean
            
            # Update paths
            # Need C^{-1/2}
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C_sqrt_inv = eigvecs @ np.diag(1 / np.sqrt(eigvals)) @ eigvecs.T
            
            # pc update
            pc = (1 - cs) * pc + cs * np.sqrt(mueff) * (C_sqrt_inv @ (ymean / sigma))
            
            # ps update
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_sqrt_inv @ (ymean / sigma))
            
            # Sigma update
            sigma = sigma * np.exp(cs / ds * (np.linalg.norm(ps) / chiN - 1))
            
            # C update
            C = (1 - c1 - cmu) * C
            C += c1 * (np.outer(pc, pc) + (1 - (1 - cs)**2) * mueff * np.eye(dim))
            
            # Rank-mu update
            # Sum w_i * (y_i - x_old)(y_i - x_old)^T
            # Note: y_i are the sorted offspring.
            # We need (y_i - x_old), not (y_i - x_new).
            for i in range(mu):
                y_diff = ys_sorted[i] - x_old
                C += cmu * w[i] * np.outer(y_diff, y_diff)
                
            # Symmetrize and ensure PD
            C = (C + C.T) / 2
            # If C becomes singular, reset to identity?
            # Check condition number
            # For robustness, if Cholesky fails next time, we reset C.
            
            # Clamp sigma
            if sigma > 10.0: sigma = 10.0
            if sigma < 1e-6: sigma = 1e-6

        return best_x, best_f
