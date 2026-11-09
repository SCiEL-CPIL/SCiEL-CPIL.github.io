import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        # CMA-ES Parameters
        n = dim
        lam = 4 + int(3 * np.log(n))  # ~9 for n=5
        mu = lam // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mueff = int(1.0 / np.sum(w**2))
        
        # Adaptation parameters
        cs = (mueff + 1.7) / (n + mueff + 5)
        ds = 1 + 2 * max(0, (mueff - 1) / (n + 1)) + cs
        cmu = (2 * (mueff - 2 + 1.0 / mueff) - 1) / ((n + 2)**2 + mueff)
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c1 = 2 / ((n + 1.3)**2 + mueff)
        cmu1 = min(1 - c1, 2 * (mueff - 2 + 1.0 / mueff) / ((n + 2)**2 + mueff))
        
        # Initialization
        xmean = np.zeros(n)
        sigma = 2.0
        C = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1/(4*n) + 1/(21*n**2))
        
        best_x = xmean.copy()
        best_f = func(xmean)
        calls = 1
        
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Generate population
            ys = np.zeros((lam, n))
            xs = np.zeros((lam, n))
            fs = np.zeros(lam)
            
            for i in range(lam):
                if calls >= budget:
                    break
                z = np.random.standard_normal(n)
                y = B @ (D * z)
                x = xmean + sigma * y
                x = np.clip(x, -5, 5)
                ys[i] = y
                xs[i] = x
                fs[i] = func(x)
                calls += 1
                
                if fs[i] < best_f:
                    best_f = fs[i]
                    best_x = x.copy()
                    
            # If we didn't fill the population, break
            if calls < budget and np.isnan(fs).any():
                break
            if calls >= budget:
                # Process the partial population? 
                # CMA-ES update requires full population.
                # If we have partial, we can just stop.
                break

            # Sort by fitness
            idx = np.argsort(fs)
            ys = ys[idx]
            xs = xs[idx]
            fs = fs[idx]
            
            # Update mean
            xmean_old = xmean.copy()
            xmean = np.sum(ys[:mu] * w[:, np.newaxis], axis=0) * sigma + xmean_old # Wait, ys are in z-space?
            # No, standard CMA-ES:
            # x = xmean + sigma * B * D * z
            # So y = B * D * z
            # xmean_new = xmean_old + sigma * (B * D * z_mean)
            # z_mean = sum(w_i * z_i)
            # But we stored ys = B * D * z. So y_mean = sum(w_i * y_i).
            # xmean_new = xmean_old + sigma * y_mean
            y_mean = np.sum(ys[:mu] * w[:, np.newaxis], axis=0)
            xmean = xmean_old + sigma * y_mean
            
            # Update evolution paths
            # pc = (1-cs)pc + cs * sqrt(mueff) * (xmean - xmean_old) / sigma
            pc = (1 - cs) * pc + cs * np.sqrt(mueff) * y_mean
            
            # ps = (1-ds)ps + sqrt(cs(2-cs)mueff) * C^{-1/2} (xmean - xmean_old) / sigma
            # C^{-1/2} = B^T D^{-1}
            # (xmean - xmean_old) / sigma = y_mean
            # z_mean = D^{-1} B^T y_mean
            z_mean = (D**(-1)) * (B.T @ y_mean)
            ps = (1 - ds) * ps + np.sqrt(cs * (2 - cs) * mueff) * z_mean
            
            # Update covariance matrix
            delta_sig = np.exp(cs * (np.linalg.norm(ps) / chi_n - 1)) + 1
            delta_sig = min(1 / cs, max(1e-3, delta_sig))
            
            rank_one = np.outer(ps, ps)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                rank_mu += w[i] * np.outer(ys[i], ys[i])
                
            C = (1 + c1 * (delta_sig - 1)) * C + c1 * (np.outer(pc, pc) - C) + cmu1 * (rank_one * delta_sig + rank_mu)
            
            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chi_n - 1))
            
            # Decompose C
            D2, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D2, 1e-12))
            # Ensure B is orthogonal (eigh returns orthogonal eigenvectors)
            
        return best_x, best_f
