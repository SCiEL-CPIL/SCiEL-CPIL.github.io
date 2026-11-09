import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Deadline guard: The evaluator kills at 330s. 
        # We aim to finish within 250s to be safe.
        start_time = time.time()
        deadline = start_time + 250.0
        
        # CMA-ES Parameters
        # For dim=5, a population size of 4 * (4 + log(5)) ~ 22 is standard, 
        # but for small budgets and small dims, a smaller population can be more efficient.
        # Let's use lambda = 9 (4 + 3*log(5) approx) or just a fixed small number.
        # Standard CMA-ES suggests lambda = 4 + floor(3 * log(n))
        lam = int(4 + 3 * np.log(dim))
        if lam < 4: lam = 4
        
        # Initial parameters
        x = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        # Weights
        mu = lam // 2
        weights = np.log(lam + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights[:mu])
        
        # Step sizes for adaptation
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))
        
        defit = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        
        # History
        best_x = x
        best_f = float('inf')
        
        # Initial evaluation
        # Start at 0,0... is often good for BBOB if shift is not huge, 
        # but a random start is safer against hidden shifts.
        # Let's start at 0 for simplicity, CMA-ES will move.
        # Actually, random start in [-5, 5] is robust.
        rng = np.random.default_rng(42)
        x = rng.uniform(-5, 5, dim)
        
        calls = 0
        while calls < budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate population
            pop = np.zeros((lam, dim))
            fitness = np.zeros(lam)
            
            for i in range(lam):
                # Sample
                # Use Cholesky decomposition for C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback if C becomes singular (rare but possible)
                    C = np.eye(dim)
                    L = np.eye(dim)
                    
                z = rng.standard_normal(dim)
                y = x + sigma * (L @ z)
                # Clip to box
                y = np.clip(y, -5, 5)
                
                pop[i] = y
                fitness[i] = func(y)
                calls += 1
                
                if fitness[i] < best_f:
                    best_f = fitness[i]
                    best_x = y.copy()
                    
            # Sort by fitness
            idx = np.argsort(fitness)
            fitness = fitness[idx]
            pop = pop[idx]
            
            # Update mean
            w = weights[:mu]
            x_new = np.sum(w[:, None] * pop[:mu], axis=0)
            
            # Update step size control variables
            # ps update
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (L.T @ ((x_new - x) / sigma))
            
            # hsig check
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (calls / (dim + 1))) < (1 + 2 / (dim + 1)) * np.sqrt(dim + 1 - 1)
            
            # Update covariance
            if hsig:
                pc = (1 - cc) * pc + cc * (x_new - x) / sigma
            
            delta_mu = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
            C = (1 + c1 * delta_mu - c1 - cmu) * C
            C += c1 * np.outer(pc, pc)
            C += cmu * np.sum(w[:, None, None] * np.outer(pop[:mu] - x, pop[:mu] - x), axis=0)
            
            # Update sigma
            sigma *= np.exp(cs / defit * (np.linalg.norm(ps) / np.sqrt(dim) - 1))
            
            # Update mean
            x = x_new
            
            # Prevent sigma from becoming too small or too large
            if sigma < 1e-10:
                break
            if sigma > 10:
                sigma = 10

        return best_x, best_f
