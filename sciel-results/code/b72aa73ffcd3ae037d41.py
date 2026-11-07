import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # Safety margin

        # CMA-ES Parameters
        # For dim=5, lam=10 is standard and efficient
        lam = int(4 + 3 * np.log(dim))
        if lam < 2:
            lam = 2
        mu = int(lam / 2)
        
        # Weights
        weights = np.log(np.arange(1, mu + 1)) - np.log(mu)
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Learning rates
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = (2 - mueff * cs + cs) / ((dim + 2) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        chiN = np.sqrt(dim) * (1 - 1. / (4 * dim) + 1. / (21 * dim ** 2))
        
        # Initialization
        rng = np.random.default_rng(42)
        x0 = rng.uniform(-5, 5, dim)
        sigma = 1.0
        C = np.eye(dim)
        
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        best_x = x0
        best_f = func(best_x)
        
        call_count = 1
        
        while call_count + lam <= budget:
            if time.time() > deadline:
                break
                
            # 1. Sample population
            # Use Cholesky decomposition of C for sampling
            try:
                Cholesky = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = C + 1e-9 * np.eye(dim)
                Cholesky = np.linalg.cholesky(C)
                
            ys = np.array([Cholesky @ z for z in rng.standard_normal((lam, dim))])
            xs = np.array([np.clip(x0 + sigma * y, -5, 5) for y in ys])
            
            # 2. Evaluate
            fs = np.array([func(x) for x in xs])
            call_count += lam
            
            # 3. Sort by fitness
            idx = np.argsort(fs)
            xs = xs[idx]
            fs = fs[idx]
            ys = ys[idx]
            
            # Update best
            if fs[0] < best_f:
                best_f = fs[0]
                best_x = xs[0]
            
            # 4. Update Mean
            xold = x0
            x0 = np.dot(weights, xs[:mu])
            
            # 5. Update Step Size
            ys_mean = np.dot(weights, ys[:mu])
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / chiN * Cholesky @ ys_mean
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / chiN - 1))
            
            # 6. Update Covariance
            # hsig: Heaviside function for rank-1 update
            # hs: Heaviside function for pc update
            hs = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (call_count) / dim)) < (1.4 + 2 / (dim + 1)) * chiN
            hsig = 1 if hs else 0
            
            cc = 2 * (mueff - 2 + 1 / mueff) / (dim ** 2 + 2)
            pc = (1 - cc) * pc + hsig * cc * (x0 - xold) / sigma
            
            # Rank-mu update
            ys_mu = ys[:mu] / sigma
            ysm = np.mean(ys_mu, axis=0)
            rank_mu_term = np.zeros((dim, dim))
            for i in range(mu):
                rank_mu_term += weights[i] * np.outer(ys_mu[i] - ysm, ys_mu[i] - ysm)
                
            # Rank-1 update
            rank_1_term = np.outer(ps, ps)
            
            # Update C
            C = (1 + c1 * hsig - c1 - cmu) * C + c1 * rank_1_term + cmu * rank_mu_term
            
            # Regularize C to ensure positive definiteness
            # Ensure symmetry
            C = (C + C.T) / 2
            
        return best_x, best_f
