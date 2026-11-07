import numpy as np
import time

class Optimizer:
    """CMA-ES optimizer for the Bent Cigar function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size lambda
        lam = 4 + int(3 * np.log(dim))
        # Number of selected individuals
        mu = lam // 2
        
        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        # Variance-effective population size
        mueff = 1.0 / np.sum(weights**2)
        
        # Initial state
        x = np.zeros(dim)  # Start at origin, safe for [-4, 4]^5 minimizer
        sigma = 2.0        # Initial step size, covers most of [-5, 5]
        
        # Covariance matrix
        C = np.eye(dim)
        # Square root of C for sampling
        C_root = np.eye(dim)
        # Inverse square root of C for updates
        C_inv_root = np.eye(dim)
        
        # Step size control
        cs = (mueff + 2) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, (mueff - 2) / (dim + 2)) + cs
        hsig = (1 + 0.3 * (mueff / dim) + cs) * np.sqrt((1 - cs) / (1 + cs))
        hsig = 1 - hsig
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Best solution found
        x_best = x.copy()
        f_best = func(x_best)
        
        start_time = time.time()
        eval_count = 1
        
        # Main loop
        while eval_count < budget:
            # Check time limit (safety margin of 5 seconds)
            if time.time() - start_time > 295:
                break
            
            # Generate population
            ys = []
            fs = []
            for i in range(lam):
                # Sample from N(x, sigma^2 C)
                # Using C_root for stability
                z = np.random.randn(dim)
                y = x + sigma * (C_root @ z)
                
                # Clip to box [-5, 5]
                y = np.clip(y, -5, 5)
                
                # Evaluate
                fy = func(y)
                eval_count += 1
                
                if fy < f_best:
                    f_best = fy
                    x_best = y.copy()
                
                ys.append(y)
                fs.append(fy)
                
                # Early exit if budget is reached
                if eval_count >= budget:
                    break
            
            if eval_count >= budget:
                break
            
            # Sort population by fitness
            sort_idx = np.argsort(fs)
            ys = [ys[i] for i in sort_idx[:mu]]
            fs = [fs[i] for i in sort_idx[:mu]]
            
            # Mean update
            y_mean = sum(w * y for w, y in zip(weights, ys))
            x_new = x + (y_mean - x)
            
            # Update sigma
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) / sigma * (y_mean - x)
            sigma *= np.exp(cs / ds * (np.linalg.norm(p_sigma) / c_norm(dim) - 1))
            
            # Update covariance matrix
            p_c = (1 - cs) * p_c + np.sqrt(cs * (2 - cs) * mueff) * (y_mean - x) / sigma
            
            # Check for h_c
            if np.linalg.norm(p_c) / np.sqrt(1 - (1 - cs)**2) < (1.4 + 2 / (dim + 1)) * c_norm(dim):
                h_c = 1
            else:
                h_c = 0
            
            C = (1 + mu * (1 - h_c) * cs * (2 - cs) - (1 - cs)) * C + \
                (1 + h_c * cs * (2 - cs) - (1 - cs)) * np.outer(p_c, p_c)
            
            # Update C_root and C_inv_root
            # Simple update: recompute every few generations or use online update
            # For simplicity and robustness, we recompute the factorization
            # This is computationally expensive but dim=5 is small
            C = 0.5 * (C + C.T)  # Symmetrize
            eigvals, eigvecs = np.linalg.eigh(C)
            # Clip eigenvalues to positive
            eigvals = np.clip(eigvals, 1e-12, None)
            C_root = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
            C_inv_root = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            
            # Update mean
            x = x_new
            
            # Ensure x is within bounds (though CMA-ES handles this implicitly via sampling, 
            # keeping the mean inside helps)
            x = np.clip(x, -5, 5)
            
        return x_best, f_best

def c_norm(dim):
    """Expected norm of a standard normal vector in dim dimensions."""
    return np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
