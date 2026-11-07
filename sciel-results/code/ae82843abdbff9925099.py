"""
Evolved Optimizer for BBOB f20 (Schwefel) using CMA-ES.
Replaces the previous population-based heuristic with a standard, robust
CMA-ES implementation. CMA-ES is specifically designed for continuous
non-convex optimization and adapts the step size and the shape of the
search distribution (covariance matrix) to the landscape, allowing it to
navigate the shifted and rotated minima of the BBOB instances more
effectively than simple random search or basic evolutionary strategies.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0
        
        calls = 0
        
        def can_continue():
            return calls < budget and time.time() < deadline

        def make_func_counter():
            nonlocal calls
            def f(x):
                nonlocal calls
                calls += 1
                val = func(x)
                if not np.isfinite(val):
                    return 1e10
                return val
            return f

        f_count = make_func_counter()
        
        # CMA-ES parameters
        # Standard lambda for CMA-ES
        lam = 4 + int(3 * np.log(dim))
        mu = lam // 2
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / np.sum(w)
        mu_eff = 1.0 / np.sum(w**2)
        
        # Learning rates
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * ((mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff)))
        
        # Initial sigma (step size)
        # For Schwefel, the optimal value is often far away. 
        # A reasonably large initial step helps exploration.
        sigma = 2.0 
        
        # Initial mean
        # Start at 0, which is usually a reasonable center for [-5, 5]^d
        x_mean = np.zeros(dim)
        
        # Covariance matrix
        C = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Precompute Cholesky factorization
        def cholesky(mat):
            try:
                return np.linalg.cholesky(mat)
            except np.linalg.LinAlgError:
                # Add small regularization if not positive definite
                return np.linalg.cholesky(mat + 1e-9 * np.eye(dim))
        
        B = cholesky(C)
        D = np.sqrt(np.diag(C))
        inv_D = 1.0 / D
        B_inv_D = np.linalg.inv(B)
        
        best_x = x_mean.copy()
        best_f = f_count(best_x)
        
        # Main loop
        while can_continue():
            # Generate population
            pop_x = []
            for i in range(lam):
                z = np.random.randn(dim)
                x = x_mean + sigma * B @ (D * z)
                # Clip to bounds [-5, 5]
                x = np.clip(x, -5, 5)
                pop_x.append(x)
            
            # Evaluate
            pop_f = [f_count(x) for x in pop_x]
            
            # Sort by fitness
            indices = np.argsort(pop_f)
            sorted_x = [pop_x[i] for i in indices]
            sorted_f = [pop_f[i] for i in indices]
            
            # Update best
            if sorted_f[0] < best_f:
                best_f = sorted_f[0]
                best_x = sorted_x[0].copy()
                
            # Selection
            y = np.array([sorted_x[i] for i in range(mu)])
            
            # Weighted mean of best mu
            y_mean = np.sum(w[:, np.newaxis] * y, axis=0)
            
            # Update p_sigma
            hs = np.linalg.norm(y_mean) / np.sqrt(1 - (1 - 0.5**((calls + 1) / (cs * (dim + 1))))) / np.linalg.norm(B @ (D * np.ones(dim)))
            if hs < 1.4 + 2 / (dim + 1):
                p_sigma = (1 - cs) * p_sigma + cs * hs * y_mean / sigma
            
            # Update p_c
            p_c = (1 - cc) * p_c + cc * hs * y_mean / sigma
            
            # Update C
            C = (1 + c1 * cmu * (mu_eff - 2 + 1 / mu_eff) - c1 - cmu) * C \
                + c1 * np.outer(p_c, p_c) \
                + cmu * np.sum(w[:, np.newaxis, np.newaxis] * np.outer(y, y), axis=0)
                
            # Note: The standard CMA-ES update for C is slightly more complex with the rank-one and rank-mu updates.
            # A simplified version often used:
            # C = (1 - c1 - cmu) * C + c1 * (p_c @ p_c.T + (1 - (1-cc)^2) * C) + cmu * sum(w_i * (y_i - y_mean) @ (y_i - y_mean).T)
            
            # Let's use the standard robust update:
            # Re-calculate C update properly
            C = (1 - c1 - cmu) * C 
            C += c1 * (np.outer(p_c, p_c) + (1 - (1 - cc)**2) * C)
            for i in range(mu):
                C += w[i] * np.outer(y[i] - y_mean, y[i] - y_mean)
            
            # Update x_mean
            x_mean = x_mean + sigma * y_mean
            
            # Update sigma
            sigma = sigma * np.exp(cs * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Ensure sigma doesn't become too small or large
            sigma = np.clip(sigma, 1e-6, 10.0)
            
            # Update Cholesky
            B = cholesky(C)
            D = np.sqrt(np.diag(C))
            
            # Early stopping check (optional, but good for efficiency)
            # If no improvement for a long time, we could restart, but with 500 budget,
            # one run is usually sufficient if CMA-ES is well tuned.
            
        return best_x, best_f

# EVOLVE-BLOCK-END
