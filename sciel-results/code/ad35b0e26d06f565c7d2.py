"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        rng = np.random.default_rng(0)
        
        # Initial population: 5 points
        # Start near center as it's often safer for shifted/rotated problems, 
        # but spread out enough to capture the landscape.
        pop_size = 5
        x0 = np.zeros(self.dim)
        
        # Initial CMA-ES parameters
        # Standard CMA-ES parameters
        lambda_ = 4 * int(np.floor(1 + 3 * np.log(self.dim)))  # Population size
        mu = lambda_ // 2  # Number of parents
        weights = np.log(np.arange(mu, 0, -1)) - np.log(mu)
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        cc = (4 + mueff / self.dim) / (self.dim + 4 + 2 * mueff / self.dim)
        cs = (mueff + 2) / (self.dim + mueff + 5)
        c1 = 2 / ((self.dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((self.dim + 2) ** 2 + mueff))
        
        # Step size
        sigma = 2.0
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        C = np.eye(self.dim)
        Cinv_sqrt = np.eye(self.dim)
        # Cholesky factor of C (initially identity)
        # We use a simplified update or direct matrix ops for small dim=5
        # For dim=5, direct matrix inversion/sqrt is fast enough.
        
        def cholesky_inverse_sqrt(A):
            try:
                L = np.linalg.cholesky(A)
                # A = L L^T
                # We want A^{-1/2}
                # Solve L X = I, then X L^T = A^{-1/2}
                X = np.linalg.solve(L, np.eye(self.dim))
                return np.linalg.solve(L.T, X.T).T
            except np.linalg.LinAlgError:
                # Fallback to eigen if Cholesky fails (numerical issues)
                w, v = np.linalg.eigh(A)
                w = np.clip(w, 1e-12, None)
                return v @ np.diag(1.0 / np.sqrt(w)) @ v.T

        # Initial population generation
        x_pop = []
        f_pop = []
        
        # Generate initial lambda_ points
        # x0 + sigma * N(0, C)
        # Since C=I initially, just x0 + sigma * N(0,1)
        
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        # Pre-generate initial population to ensure we have enough calls
        # Budget is 500. Lambda is approx 4*(1+3*log(5)) = 4*(1+4.8) = 23.
        # 500 / 23 ~ 21 generations.
        
        for gen in range(25):  # Max generations cap
            if time.time() > deadline - 5.0:
                break
                
            # Generate population
            x_pop = []
            f_pop = []
            
            for i in range(lambda_):
                if time.time() > deadline - 2.0:
                    break
                if calls_made >= self.budget:
                    break
                    
                # Sample
                # x = x0 + sigma * (Cinv_sqrt @ z)
                z = rng.standard_normal(self.dim)
                x = x0 + sigma * (Cinv_sqrt @ z)
                x = np.clip(x, -5, 5)
                
                f = func(x)
                calls_made += 1
                
                x_pop.append(x)
                f_pop.append(f)
                
                if f < best_f:
                    best_f = f
                    best_x = x
            
            if not x_pop:
                break
                
            # Sort
            indices = np.argsort(f_pop)
            x_pop = np.array(x_pop)[indices]
            f_pop = np.array(f_pop)[indices]
            
            # Update x0 (mean of parents)
            x_parents = x_pop[:mu]
            x_new = np.sum(weights[:, np.newaxis] * x_parents, axis=0)
            
            # Update evolution paths
            pc = (1 - cc) * pc + cc / np.sqrt(cc) * Cinv_sqrt @ (x_new - x0)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * Cinv_sqrt @ (x_new - x0)
            
            # Update sigma
            sigma *= np.exp(cs / (self.dim + 2) * (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) - np.sqrt(self.dim * (1 - 1 / self.dim))))
            
            # Update covariance matrix
            diff = x_parents - x0
            delta_h = (1 / np.sqrt(cc)) * (1 - np.exp(-2 * cc * gen / self.dim)) * (x0 - x_new)
            
            # C = (1 - c1 - cmu) * C + c1 * (pc pc^T + (1 - (1 - 2*cc) * ||pc||^2) * I) + cmu * sum(weights * diff diff^T)
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - (1 - 2 * cc) * np.dot(pc, pc)) * np.eye(self.dim))
            for i in range(mu):
                C += cmu * weights[i] * np.outer(diff[i], diff[i])
            
            # Symmetrize
            C = (C + C.T) / 2
            
            # Ensure positive definiteness
            # Simple correction if needed
            try:
                Cinv_sqrt = cholesky_inverse_sqrt(C)
            except Exception:
                # If Cholesky fails, revert to previous or identity
                C = np.eye(self.dim)
                Cinv_sqrt = np.eye(self.dim)
                sigma = max(sigma * 0.5, 0.1)
                
            x0 = x_new
            
        return best_x, best_f

# EVOLVE-BLOCK-END
