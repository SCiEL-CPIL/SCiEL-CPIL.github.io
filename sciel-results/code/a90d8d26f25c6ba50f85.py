"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES for BBOB f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Initial point
        x0 = np.zeros(dim)
        f0 = func(x0)
        best_x = x0.copy()
        best_f = f0
        evals = 1
        
        if evals >= budget:
            return best_x, best_f

        # CMA-ES parameters
        lambda_pop = 4 + int(np.floor(3 * np.log(dim)))
        mu = lambda_pop // 2
        
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        sigma = 2.0
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        
        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3)**2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))
        
        x0_prev = x0.copy()
        
        while evals < budget:
            # Time guard
            if time.time() > deadline:
                break
                
            # Generate offspring
            try:
                D, V = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-12))
                C_inv_sqrt = V @ np.diag(1.0 / D) @ V.T
            except Exception:
                break
            
            points = []
            for i in range(lambda_pop):
                z = np.random.randn(dim)
                y = x0 + sigma * (C_inv_sqrt @ z)
                y = np.clip(y, -5, 5)
                points.append(y)
            
            # Evaluate
            fvals = np.array([func(p) for p in points])
            evals += lambda_pop
            
            # Update best
            min_idx = np.argmin(fvals)
            if fvals[min_idx] < best_f:
                best_f = fvals[min_idx]
                best_x = points[min_idx].copy()
                
            # Check if budget exhausted
            if evals >= budget:
                break
            
            # Sort
            order = np.argsort(fvals)
            y_sorted = [points[i] for i in order]
            
            # Update mean
            x0_new = np.sum([weights[i] * y_sorted[i] for i in range(mu)], axis=0)
            
            # Update path evolution strategies
            pc = (1 - cc) * pc + cc * (x0_new - x0_prev) / sigma
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (x0_new - x0_prev)
            
            # Update covariance matrix
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs)**2) / chi_n) < (1.4 + 2 / (dim + 1))
            C = (1 + c1 * hsig * (cc + 1 - cc) - c1 - cmu) * C
            C += c1 * (pc @ pc.T - (1 - hsig) * (1 - cc) * C)
            
            # Sum of weighted outer products
            rank_mu_update = 0
            for i in range(mu):
                diff = (y_sorted[i] - x0_prev) / sigma
                rank_mu_update += weights[i] * np.outer(diff, diff)
            C += cmu * rank_mu_update
            
            # Update step size
            sigma = sigma * np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
            
            # Ensure sigma doesn't get too small or large
            sigma = np.clip(sigma, 1e-6, 100.0)
            
            x0_prev = x0.copy()
            x0 = x0_new
            
        return best_x, best_f
# EVOLVE-BLOCK-END
