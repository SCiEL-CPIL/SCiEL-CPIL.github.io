"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        # CMA-ES parameters
        dim = self.dim
        # Initial step size
        sigma = 2.0
        # Initial mean (center of the box)
        x = np.zeros(dim)
        
        # Population size
        lambda_pop = 4 + int(np.floor(3 * np.log(dim)))
        mu = lambda_pop // 2  # Number of selected individuals
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights ** 2)
        
        # Adaptation rates
        cc = (4. + mueff / (dim + 4.)) / ((dim + 1.7) + mueff / 2.)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        
        # State variables
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        C = np.eye(dim)
        
        # Initial evaluation
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1
        
        # Main loop
        while calls < self.budget:
            if time.time() > deadline:
                break
                
            # Generate population
            ys = np.zeros((lambda_pop, dim))
            fs = np.zeros(lambda_pop)
            
            for i in range(lambda_pop):
                # Sample from N(0, C)
                # Efficient sampling: B @ (D * z)
                z = np.random.randn(dim)
                y = x + sigma * (B @ (D * z))
                # Clip to box
                y = np.clip(y, -5, 5)
                ys[i] = y
                fs[i] = func(y)
                calls += 1
                
                if fs[i] < best_fx:
                    best_fx = fs[i]
                    best_x = y.copy()
                    
                if time.time() > deadline:
                    break
            
            if calls >= self.budget:
                break
                
            # Sort by fitness
            idx = np.argsort(fs)
            ys = ys[idx]
            fs = fs[idx]
            
            # Update mean
            x_new = np.sum(weights[:, None] * ys[:mu], axis=0)
            
            # Update step size (CSA)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / D @ C @ (x_new - x) / sigma
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls + 1) / (dim * damps))) < (1.4 + 2 / (dim + 1)) * np.sqrt(dim):
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
                
            # Update evolution path for covariance (CC)
            pc = (1 - cc) * pc + np.sqrt(cc * (2 - cc) * mueff) / D @ C @ (x_new - x) / sigma
            
            # Update covariance matrix
            delta = 0 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls + 1) / (dim * damps))) < (1.4 + 2 / (dim + 1)) * np.sqrt(dim) else 1
            rank_mu = np.outer(x_new - x, x_new - x)
            rank_one = np.outer(pc, pc)
            
            C = (1 + c1 * delta - c1 - cmu) * C + c1 * rank_one + cmu / mueff * np.sum(weights[:, None, None] * np.outer(ys[:mu] - x, ys[:mu] - x), axis=0)
            
            # Decompose C = B @ diag(D^2) @ B.T
            D2, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D2, 1e-12))
            
            x = x_new
            fx = best_fx # We track best separately, but x is the mean
            
        return best_x, best_fx
# EVOLVE-BLOCK-END
