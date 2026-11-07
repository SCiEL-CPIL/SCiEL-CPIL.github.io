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
        t0 = time.time()
        deadline = t0 + 280  # 300s limit, 20s safety margin
        
        n = self.dim
        budget = self.budget
        
        # Initial parameters
        x0 = np.zeros(n)
        f0 = func(x0)
        
        # CMA-ES parameters
        # lambda is population size
        lam = max(4 + int(3 * np.log(n)), 2)
        
        # Initial sigma
        # Bent cigar has one very narrow dimension. 
        # A standard deviation of 2.0 covers the box well initially.
        sigma = 2.0
        
        # Covariance matrix C
        C = np.eye(n)
        
        # Evolution paths
        ps = np.zeros(n)
        pc = np.zeros(n)
        
        # Learning rates
        cmu = 2 / ((n + 4) ** 2 + lam)
        c1 = (2 + 2 / (n + 1.3)) / ((n + 1) ** 2 + lam)
        cmu = 2 * (cmu - c1 / 2)
        cs = (n + 1.5) / (n + n / 2 + 5)
        c_sigma = 1 / (n + 3)
        d_sigma = 1 + 2 * max(0, np.sqrt((lam - 1) / (n + 1)) - 1) + cs
        
        # Step size control
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        
        # Initial best
        best_x = x0
        best_f = f0
        
        # Budget management
        # We have 'budget' total evaluations. One used for x0.
        # Remaining: budget - 1
        # Each generation uses 'lam' evaluations.
        
        calls_made = 1
        
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            pops = np.zeros((lam, n))
            popf = np.zeros(lam)
            
            # We might not have enough budget for a full population
            lam_cur = min(lam, budget - calls_made)
            
            for i in range(lam_cur):
                # Sample from multivariate normal
                # Using Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback if C becomes singular (rare but possible)
                    C = np.eye(n)
                    L = np.eye(n)
                
                # z ~ N(0, I)
                z = np.random.standard_normal(n)
                # y = sigma * L * z
                y = sigma * (L @ z)
                # x = x0 + y
                x_new = x0 + y
                
                # Clip to box
                x_new = np.clip(x_new, -5, 5)
                
                pops[i] = x_new
                popf[i] = func(x_new)
                calls_made += 1
                
                # Update best
                if popf[i] < best_f:
                    best_f = popf[i]
                    best_x = x_new.copy()
            
            if lam_cur == 0:
                break
            
            # Sort population by fitness
            idx = np.argsort(popf)
            pops = pops[idx]
            popf = popf[idx]
            
            # Update mean
            # Weighted sum of the best mu individuals
            mu = lam_cur // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights = weights / np.sum(weights)
            
            # New mean
            x_new_mean = np.zeros(n)
            for i in range(mu):
                x_new_mean += weights[i] * pops[i]
            
            # Update evolution paths
            # ps = (1 - cs) * ps + sqrt(cs * (2 - cs) * mueff) / sigma * (C^-1/2 * (x_new_mean - x0))
            mueff = np.sum(weights) ** 2 / np.sum(weights ** 2)
            
            # C^-1/2
            try:
                C_inv_sqrt = np.linalg.inv(np.linalg.cholesky(C))
            except np.linalg.LinAlgError:
                C_inv_sqrt = np.eye(n)
            
            diff = x_new_mean - x0
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * (C_inv_sqrt @ diff)
            
            # hsig: check if ps is within the trust region
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls_made - lam_cur))) < (1.4 + 2 / (n + 1)) * chi_n
            
            # Update pc
            if hsig:
                pc = (1 - c1) * pc + np.sqrt(c1 * (2 - c1) * mueff) * C_inv_sqrt @ diff
            else:
                pc = (1 - c1) * pc
            
            # Update covariance matrix
            # C = (1 - c1 - cmu) * C + c1 * (pc * pc.T + (1 - (1 - c1) * (1 - 2 * c1)) * I) + cmu * sum(wi * yi * yi.T)
            
            # Term 1: decay
            C = (1 - c1 - cmu) * C
            
            # Term 2: rank-one update from pc
            if hsig:
                C += c1 * (pc @ pc.T + (1 - (1 - c1) * (1 - 2 * c1)) * np.eye(n))
            else:
                C += c1 * (1 - (1 - c1) * (1 - 2 * c1)) * np.eye(n)
            
            # Term 3: rank-mu update from population
            # yi = (xi - x0) / sigma
            for i in range(mu):
                yi = (pops[i] - x0) / sigma
                C += cmu * weights[i] * (yi @ yi.T)
            
            # Ensure symmetry and positive definiteness (numerical stability)
            C = (C + C.T) / 2
            
            # Update sigma
            # sigma = sigma * exp(c_sigma / d_sigma * (||ps|| / chi_n - 1))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            
            # Clamp sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update mean
            x0 = x_new_mean
            
            # Check convergence
            # If sigma is very small or C is nearly singular, we might be stuck
            # But the budget guard is the primary stopper.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
