"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        d = self.dim
        # CMA-ES Parameters
        lam = 4 + d // 2  # 7 for d=5
        mu = lam // 2     # 3
        
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights ** 2)
        
        # Step size control parameters
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (d + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2) ** 2 + mu_eff))
        
        # Initial state
        x = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)
        pc = np.zeros(d)
        ps = np.zeros(d)
        
        # Initial evaluation
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1
        
        # Pre-compute Cholesky factor
        L = np.linalg.cholesky(C)
        
        while calls < self.budget:
            if time.time() > deadline:
                break
                
            # Generate offspring
            ys = np.zeros((lam, d))
            fvals = np.zeros(lam)
            
            for i in range(lam):
                z = np.random.randn(d)
                y = x + sigma * (L @ z)
                ys[i] = np.clip(y, -5, 5)
                fvals[i] = func(ys[i])
                calls += 1
                if fvals[i] < best_fx:
                    best_fx = fvals[i]
                    best_x = ys[i].copy()
                    
                if time.time() > deadline:
                    for j in range(i + 1, lam):
                        fvals[j] = np.inf
                    break
            
            # Sort by fitness
            idx = np.argsort(fvals)
            ys_sorted = ys[idx]
            fvals_sorted = fvals[idx]
            
            # Update mean
            x_new = np.sum([weights[i] * ys_sorted[i] for i in range(mu)], axis=0)
            
            # Update step size (Sigma)
            # z_mean = L^{-1} (x_new - x) / sigma
            try:
                z_mean = np.linalg.solve(L, (x_new - x) / sigma)
            except np.linalg.LinAlgError:
                z_mean = np.zeros(d)
                
            ps = (1 - np.sqrt(c_sigma * (2 - c_sigma) * mu_eff)) * ps + \
                 np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z_mean
            
            # Update pc
            pc = (1 - c_c) * pc + c_c * (x_new - x)
            
            # Update Covariance Matrix C
            # Delta H term
            H = np.outer(pc, pc) + (1 - (1 - c_c)**2) * c_c / (1 - c_c) * np.eye(d)
            
            # Rank-mu update
            rank_mu = np.zeros((d, d))
            for i in range(mu):
                yi = ys_sorted[i] - x
                rank_mu += weights[i] * np.outer(yi, yi)
                
            C = (1 - c_1 - c_mu) * C + c_1 * H + c_mu * rank_mu
            
            # Ensure symmetry
            C = (C + C.T) / 2
            
            # Update Cholesky factor
            try:
                L = np.linalg.cholesky(C + 1e-12 * np.eye(d))
            except np.linalg.LinAlgError:
                # Fallback
                C = np.eye(d)
                L = np.eye(d)
            
            x = x_new
            
        return best_x, best_fx
# EVOLVE-BLOCK-END
