"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy)"""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        # Time guard
        start_time = time.time()
        deadline = start_time + 290.0  # 300s limit, keep 10s margin

        # Initialize CMA-ES parameters
        # x0: Start at origin (center of box), as the minimizer is in [-4,4]^5
        x0 = np.zeros(self.dim)
        sigma0 = 2.0
        
        # Population size
        lambda_ = 4 + int(np.floor(3 * np.log(self.dim)))
        mu = lambda_ // 2
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1 / np.sum(weights**2)
        
        # Learning rates
        cc = (4 + mueff / self.dim) / (self.dim + 4 + 2 * mueff / self.dim)
        cs = (mueff + 2) / (self.dim + mueff + 5)
        c1 = 2 / ((self.dim + 1.3)**2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((self.dim + 2)**2 + mueff))
        
        # State variables
        x = x0.copy()
        sigma = sigma0
        C = np.eye(self.dim)
        pc = np.zeros(self.dim)
        ps = np.zeros(self.dim)
        # Precompute sqrt of C for sampling
        # We will update C and recompute sqrt(C) each generation
        
        # Initial evaluation
        fx = func(x0)
        best_x = x0.copy()
        best_fx = fx
        
        # Evaluation counter
        calls = 1
        
        # Main loop
        while calls < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Compute sqrt(C)
            # Use eigendecomposition for stability
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip eigenvalues to positive
            eigvals = np.maximum(eigvals, 1e-12)
            sqrtC = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            
            # Sample population
            pop = np.zeros((lambda_, self.dim))
            fvals = np.zeros(lambda_)
            
            for i in range(lambda_):
                # Check time inside sampling loop if lambda is large, but lambda is small
                z = np.random.randn(self.dim)
                xi = x + sigma * (sqrtC @ z)
                xi = np.clip(xi, -5, 5)
                pop[i] = xi
                fvals[i] = func(xi)
                calls += 1
                
                # Early break if time is tight
                if time.time() > deadline:
                    # Fill remaining with best so far to avoid errors, though we break anyway
                    # Actually just break out of sampling
                    break
            
            if time.time() > deadline:
                break
            
            # Sort by fitness
            idx = np.argsort(fvals)
            pop = pop[idx]
            fvals = fvals[idx]
            
            # Update best
            if fvals[0] < best_fx:
                best_fx = fvals[0]
                best_x = pop[0].copy()
            
            # Weighted mean
            m = np.sum(weights[:, np.newaxis] * pop[:mu], axis=0)
            
            # Update step size control vector ps
            # Need C^(-1/2)
            inv_sqrtC = np.linalg.inv(sqrtC)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (inv_sqrtC @ (m - x))
            
            # Update covariance matrix
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**(2 * (calls / lambda_))) < (1.4 + 2 / (self.dim + 1)) * np.sqrt(self.dim + 1 - 2))
            
            # Update C
            # h1 = hsig * (1 - cc)
            # C = (1 - c1 - cmu + h1*cc*(cc+2*cs)) * C + c1 * (pc @ pc.T) + cmu * sum(weights_i * (y_i @ y_i.T))
            
            # Compute pc
            pc = (1 - cc) * pc + hsig * cc * (m - x)
            
            # Compute y vectors (in x-space, not z-space)
            # y_i = (x_i - x_old) / sigma
            # But we need them in the standard normal space for the update?
            # Standard CMA-ES update:
            # C <- (1 - c1 - cmu) C + c1 pc pc^T + cmu sum wi (xi - x) (xi - x)^T / sigma^2
            # Wait, the term is usually in the z-space or scaled?
            # Let's stick to the standard formulation:
            # C <- (1 - c1 - cmu + c1*h1*cc*(cc+2*cs)) C + c1 pc pc^T + cmu sum wi yi yi^T
            # where yi = (xi - x) / sigma
            
            y = (pop[:mu] - x) / sigma
            delta_C = c1 * np.outer(pc, pc)
            for i in range(mu):
                delta_C += cmu * weights[i] * np.outer(y[i], y[i])
            
            # Update C
            C = (1 - c1 - cmu + (c1 * hsig * cc * (cc + 2 * cs) if hsig else 0)) * C + delta_C
            
            # Update x
            x = m
            
            # Update sigma
            chiN = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim**2))
            sigma = sigma * np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1))
            
            # Ensure sigma doesn't explode or vanish
            sigma = np.clip(sigma, 1e-12, 10.0)
            
            # Symmetrize C
            C = (C + C.T) / 2
            
            # If calls are exhausted, break
            if calls >= self.budget:
                break
                
        return best_x, best_fx
# EVOLVE-BLOCK-END
