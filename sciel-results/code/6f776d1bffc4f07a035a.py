"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for the Bent Cigar function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin
        
        n = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        lambda_ = 4 + int(3 * np.log(n))  # 10 for n=5
        mu = lambda_ // 2  # 5
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        cs = (mueff + 1) / (n + mueff + 5)
        ds = 1 + 2 * max(0, (mueff - 1) / (n + 1)) - cs
        hsig_threshold = (1.7 + 2.0 / (n + 1)) * (1 - cs)
        
        c1 = (2 / ((n + 1.3)**2 + mueff)) * (1 - cs) * (1 + ds)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1/mueff) / ((n + 2)**2 + mueff))
        
        # Initial values
        x = np.zeros(n)
        sigma = 1.0
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        
        x_best = x.copy()
        fx_best = func(x_best)
        calls = 1
        
        # Main loop
        while calls + lambda_ <= budget:
            if time.time() > deadline:
                break
                
            # Generate population
            ys = []
            fs = []
            try:
                L = np.linalg.cholesky(C)
            except:
                C = C + 1e-8 * np.eye(n)
                L = np.linalg.cholesky(C)
                
            for i in range(lambda_):
                z = np.random.standard_normal(n)
                y = x + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls += 1
                
                if calls > budget:
                    break
                
                ys.append(y)
                fs.append(fy)
                
                if fy < fx_best:
                    fx_best = fy
                    x_best = y.copy()
            
            if len(fs) < lambda_:
                break
                
            # Sort
            indices = np.argsort(fs)
            ys = [ys[i] for i in indices]
            fs = [fs[i] for i in indices]
            
            # Update mean
            yw = np.sum([weights[i] * ys[i] for i in range(mu)], axis=0)
            x = x + yw
            
            # Update ps
            try:
                C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C))
            except:
                C_inv_sqrt = np.eye(n)
                
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_inv_sqrt @ yw) / sigma
            
            # Update pc
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) / np.sqrt(1 - (1 - cs)**(2 * (calls // lambda_))) < hsig_threshold
            pc = (1 - cs) * pc + hsig * cs * yw / sigma
            
            # Update C
            C = (1 + c1 * (2 - cs) * (hsig - 1) + c1 * ds) * C \
                - c1 * (np.outer(pc, pc) - C) \
                + cmu * np.sum([weights[i] * np.outer(ys[i] - yw, ys[i] - yw) for i in range(mu)], axis=0)
            
            C = (C + C.T) / 2
            try:
                np.linalg.cholesky(C)
            except:
                C = C + 1e-6 * np.eye(n)
                
        return x_best, fx_best
# EVOLVE-BLOCK-END
