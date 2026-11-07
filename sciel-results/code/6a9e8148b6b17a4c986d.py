import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size
        lam = max(4, 4 + int(3 * np.log(dim)))
        # Ensure we can do at least a few generations
        if lam > budget // 3:
            lam = max(4, budget // 3)
            
        mu = lam // 2  # Number of parents
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Effective population size
        mueff = 1.0 / np.sum(weights**2)
        
        # Step size control parameters
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        chi = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Covariance matrix adaptation parameters
        c_c = (4 + mueff/dim) / (dim + 4 + 2*mueff/dim)
        c_1 = 2 / ((dim + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1/mueff) / ((dim + 2)**2 + mueff))
        
        # Initial state
        x = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        C_sqrt = np.eye(dim)
        
        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Best found
        best_x = x.copy()
        best_f = func(best_x)
        calls = 1
        
        # Pre-compute C_sqrt_inv for path updates? 
        # Actually, we need to update C_sqrt each generation.
        
        while calls + lam <= budget and time.time() < deadline:
            # Generate population
            Y = []
            for i in range(lam):
                z = np.random.standard_normal(dim)
                y = x + sigma * C_sqrt @ z
                Y.append(y)
                
            # Evaluate population
            F = []
            X = []
            for y in Y:
                # Clip to box
                yc = np.clip(y, -5, 5)
                f = func(yc)
                calls += 1
                F.append(f)
                X.append(yc)
                
                # Update best
                if f < best_f:
                    best_f = f
                    best_x = yc.copy()
                    
            F = np.array(F)
            X = np.array(X)
            
            # Sort by fitness
            idx = np.argsort(F)
            X_sorted = X[idx]
            F_sorted = F[idx]
            
            # Select parents
            X_parents = X_sorted[:mu]
            
            # Weighted mean
            m = np.dot(weights, X_parents)
            
            # Update evolution paths
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * C_sqrt_inv * (m - x)
            # We need C_sqrt_inv. Since C_sqrt is from Cholesky, C = C_sqrt @ C_sqrt.T
            # C^{-1} = C_sqrt^{-T} @ C_sqrt^{-1}
            # This is expensive. Standard CMA-ES uses C^{-1/2} or approximations.
            # For simplicity and speed in Python, we can use the inverse of C_sqrt.
            try:
                C_sqrt_inv = np.linalg.inv(C_sqrt)
            except np.linalg.LinAlgError:
                C_sqrt_inv = np.eye(dim)
                
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (C_sqrt_inv @ (m - x))
            
            # Check if ps is within confidence interval
            if np.linalg.norm(ps) / chi < (1.4 + 2/(dim + 1)) * np.sqrt((1 - (1 - c_sigma)**(2*(budget - calls)))):
                # Update pc
                pc = (1 - c_c) * pc + c_c * (m - x) / sigma
                # Update C
                v = pc / sigma + np.sqrt(c_c * (2 - c_c) * mueff) / sigma * C_sqrt_inv @ (m - x)
                C += (c_1 + c_mu) * (np.outer(v, v) - C)
            else:
                C += c_1 * (np.eye(dim) - C)
                
            # Rank-mu update
            # For simplicity, we might skip rank-mu update or do a simple version
            # C += c_mu * sum(weights_i * outer(x_i - m, x_i - m))
            # This is expensive. Let's do a simplified version or skip it for speed.
            # Given dim=5, it's manageable.
            rank_mu_term = np.zeros((dim, dim))
            for i in range(mu):
                diff = X_parents[i] - m
                rank_mu_term += weights[i] * np.outer(diff, diff)
            C += c_mu * rank_mu_term
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add small regularization
                C += 1e-10 * np.eye(dim)
                C = (C + C.T) / 2
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C_sqrt = np.eye(dim)
                    
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi - 1))
            
            # Update mean
            x = m
            
            # Safety: if sigma becomes too small or too large, adjust
            if sigma < 1e-10:
                sigma = 1e-10
            if sigma > 100:
                sigma = 100
                
            # Check for stagnation
            if np.all(F_sorted[:mu] == F_sorted[0]):
                # All parents have same fitness, random restart or break
                # For simplicity, just continue
                pass
                
        return best_x, best_f
