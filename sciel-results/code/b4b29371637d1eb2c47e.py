import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, keep 10s margin
        
        d = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lam = 10  # Fixed population size for d=5, good balance
        mu = lam // 2
        
        # Initial parameters
        x0 = np.zeros(d) 
        sigma = 2.0      
        
        # CMA-ES weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights ** 2)
        
        # Learning rates
        cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        cs = (mueff + 2) / (d + mueff + 5)
        c1 = 2 / ((d + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((d + 2) ** 2 + mueff))
        
        # State variables
        mean = x0.copy()
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))
        
        best_x = x0.copy()
        best_f = func(x0)
        evals = 1
        
        if evals >= budget:
            return best_x, best_f

        # Main Loop
        while evals < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            samples = np.zeros((lam, d))
            for i in range(lam):
                # Sample from N(mean, sigma^2 C)
                # Use Cholesky
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback: regularize C
                    C += 1e-12 * np.eye(d)
                    L = np.linalg.cholesky(C)
                
                z = np.random.randn(d)
                samples[i] = mean + sigma * (L @ z)
                samples[i] = np.clip(samples[i], -5, 5)
            
            # Evaluate
            fvals = np.array([func(samples[i]) for i in range(lam)])
            evals += lam
            
            # Update best
            idx_min = np.argmin(fvals)
            if fvals[idx_min] < best_f:
                best_f = fvals[idx_min]
                best_x = samples[idx_min].copy()
            
            # Sort
            order = np.argsort(fvals)
            x_sorted = samples[order]
            
            # Update mean
            mean_old = mean.copy()
            mean = np.sum(weights[:, None] * x_sorted[:mu], axis=0)
            
            # Update ps
            # Need C^{-1/2}
            try:
                C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C))
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C + 1e-12 * np.eye(d)))
                
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_inv_sqrt @ (mean - mean_old) / sigma)
            
            # Update pc
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (evals / lam + 1)))) < (1 + 1.4 / d) * chiN
            if hsig:
                pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) * (mean - mean_old) / sigma
            
            # Update Covariance Matrix C
            diff = (x_sorted[:mu] - mean_old) / sigma
            rank_mu = (weights[:, None] * diff).T @ diff
            rank_one = np.outer(pc, pc)
            
            delta = 1 + 2 * max(0, (hsig - 1) / (d + 1)) * c1
            C = (1 + c1 * (delta - 1) - c1 - cmu) * C + c1 * rank_one + cmu * rank_mu
            C = (C + C.T) / 2
            
            # Update Sigma
            sigma *= np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1))
            
            if sigma < 1e-12:
                sigma = 1e-12
            if sigma > 100:
                sigma = 100

        return best_x, best_f
