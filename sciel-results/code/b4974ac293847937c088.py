"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES with population size 4, starting from origin.
    Uses cumulative step-size control for robustness against hidden
    rotation and shift in BBOB f14.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290

        rng = np.random.default_rng(0)
        n = self.dim
        lam = 4  # population size
        
        # Start from origin (known locality of f14 minimum)
        x = np.zeros(n)
        fx = func(x)
        calls_made = 1
        
        # CMA-ES parameters
        sigma = 2.0
        C = np.eye(n)
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Step sizes
        c_sigma = (lam/2 + 1) / (n + 2 + lam/2)
        d_sigma = 1 + 2 * max(0, np.sqrt((lam/2 - 1)/(n + 1)) - 1) + c_sigma
        c_c = (4 + lam/n) / (n + 4 + 2*lam/n)
        c_1 = (2 + 2*min(1, (lam/2 - 1)/(n + 1))) / (n + 2)**1.8 + c_c
        c_mu = min(1 - c_1, 2 * (lam/2 - 2 + 1/(lam/2)) / (n + 2)**1.8)
        
        # Weights for mu-best selection
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        while calls_made < self.budget:
            if time.time() > deadline:
                break
            
            # Generate population
            B = np.linalg.cholesky(C)
            P = sigma * B
            
            Y = []
            F = []
            for i in range(lam):
                z = rng.standard_normal(n)
                y = x + P @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_made += 1
                Y.append(y)
                F.append(fy)
            
            Y = np.array(Y)
            F = np.array(F)
            
            # Sort by fitness
            idx = np.argsort(F)
            Y_sorted = Y[idx]
            F_sorted = F[idx]
            
            # Update best
            if F_sorted[0] < fx:
                fx = F_sorted[0]
            
            # Weighted mean of mu best
            w = weights[:mu]
            m = np.sum(w[:, None] * Y_sorted[:mu], axis=0)
            
            # Update C (covariance matrix adaptation)
            v = (m - x) / sigma
            C = (1 + c_1 * c_mu * (1 - c_1) * mu_eff) * C + \
                c_1 * (np.outer(v, v) - C) + \
                c_mu * np.sum(w[:, None, None] * np.outer(Y_sorted[:mu] - x, Y_sorted[:mu] - x) / sigma**2, axis=0)
            
            # Update step size (cumulative)
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * v
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update path
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * v
            h_sig = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) < (1.4 + 2/(n+1)) * np.sqrt(mu_eff) else 0
            C = (1 + c_1 * (c_mu * h_sig - c_c) * (1 - c_c) * mu_eff) * C + \
                c_1 * (np.outer(p_c, p_c) - C)
            
            # Update mean
            x = m
            x = np.clip(x, -5, 5)
            
            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            eigvals = np.linalg.eigvalsh(C)
            if np.any(eigvals < 1e-10):
                C = C + 1e-8 * np.eye(n)
                
        return x, fx
# EVOLVE-BLOCK-END
