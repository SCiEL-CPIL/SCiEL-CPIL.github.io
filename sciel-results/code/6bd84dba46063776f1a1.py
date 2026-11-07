"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with strict time budgeting for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0  # Safety margin before 300s hard limit
        
        # CMA-ES Parameters
        dim = self.dim
        popsize = int(4 + 3 * np.log(dim))
        sigma = 1.5
        x = np.zeros(dim)
        fx = np.inf
        
        # CMA-ES State
        C = np.eye(dim)
        Cs = np.zeros((dim, dim))
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        
        # Step sizes
        cmu = 2 / ((dim + 4)**2 + 10)
        c1 = (2 + 2 / (dim + 1)) / ((dim + 1)**2 + 2)
        cmu = min(cmu, 2 - c1)
        c1 = 2 * (1 - c1) / ((dim + 1)**2 + 2)
        
        cs = (4 + dim / 3) / (dim + 4 + 2 / dim)
        c_sigma = (4 + dim / 3) / (dim + 8)
        
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        # Initial evaluation
        x0 = np.zeros(dim)
        fx0 = func(x0)
        x, fx = x0, fx0
        
        # Adaptive step size based on function scale estimation
        # f14 typically has values in [-100, 100] or similar. 
        # We start with sigma=1.5. If fx is large, we might need to explore more, 
        # but CMA-ES adapts sigma automatically.
        
        # Precompute cholesky inverse for C
        # We will update C and recompute cholesky every step
        
        for iteration in range(self.budget):
            if time.time() > deadline:
                break
                
            # Compute Cholesky decomposition of C
            try:
                B = np.linalg.cholesky(C)
                D = np.sqrt(np.diag(C))
                minvB = np.linalg.inv(B)
            except np.linalg.LinAlgError:
                # Fallback if matrix becomes singular
                C = np.eye(dim)
                B = np.eye(dim)
                D = np.ones(dim)
                minvB = np.eye(dim)
                
            # Sample population
            Y = np.random.randn(popsize, dim)
            P = x + sigma * (Y @ B.T)
            
            # Evaluate population
            F = np.array([func(p) for p in P])
            
            # Sort by fitness
            order = np.argsort(F)
            lam = popsize
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights = weights / np.sum(weights)
            
            # Update mean
            x_new = np.sum(weights[:, None] * P[order[:mu]], axis=0)
            
            # Update covariance matrix
            # y_sigma = (x_new - x) / sigma
            y_sigma = (x_new - x) / sigma
            
            # Update path
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * B.T @ y_sigma
            hsig = 1 if (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**2) / chiN < 1.4 + 2 / (dim + 1)) else 0
            
            pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mu_eff) * y_sigma
            
            # Update C
            C = (1 - c1 - cmu) * C + c1 * (np.outer(pc, pc) + (1 - hsig) * cs * (2 - cs) * C) + cmu * np.sum(weights[:, None, None] * np.outer(P[order[:mu]] - x, P[order[:mu]] - x) / sigma**2, axis=0)
            
            # Update sigma
            sigma = sigma * np.exp(c_sigma / chiN * (np.linalg.norm(ps) / chiN - 1))
            
            # Update best if improved
            if F[order[0]] < fx:
                x, fx = P[order[0]], F[order[0]]
            else:
                x = x_new
                fx = F[order[0]] if F[order[0]] < fx else fx
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            
            # Small safety check on sigma
            if sigma < 1e-12:
                break
            if sigma > 100:
                sigma = 100

        return x, fx
# EVOLVE-BLOCK-END
