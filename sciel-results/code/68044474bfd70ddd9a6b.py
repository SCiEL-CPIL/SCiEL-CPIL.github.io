"""Optimized BBOB f14 solver with full CMA-ES."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Full CMA-ES implementation for BBOB f14."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 320.0  # Safety margin before 330s kill

        # CMA-ES Parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))  # Standard formula, for dim=5 lam=9
        # Step size
        sigma = 2.0
        # Initial mean
        # Use a few random starts to find a good initial center, as CMA-ES is local
        # But CMA-ES itself is robust if initialized reasonably.
        # Given the box [-5,5] and minimizer in [-4,4], starting at 0 is often good,
        # but let's do a tiny random search to pick the best of 3 starts to be safe.
        
        rng = np.random.default_rng(0)
        
        # Initial exploration: 3 points
        starts = [np.zeros(dim)]
        for _ in range(2):
            starts.append(rng.uniform(-4, 4, dim))
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        for s in starts:
            if time.time() > deadline:
                break
            fs = func(s)
            calls_made += 1
            if fs < best_f:
                best_f = fs
                best_x = s.copy()
                
        if best_x is None:
             best_x = np.zeros(dim)
             best_f = func(best_x)
             calls_made += 1

        x_mean = best_x.copy()
        
        # Covariance matrix C = I
        C = np.eye(dim)
        # Inverse of C
        C_inv = np.eye(dim)
        # Cholesky factor of C
        C_sqrt = np.eye(dim)
        # Inverse Cholesky
        C_inv_sqrt = np.eye(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Learning rates
        mu = lam // 2  # Selection mass
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        # Precompute
        # We need to update C_inv_sqrt and C_sqrt every generation.
        # For dim=5, this is fast.
        
        while calls_made < budget and time.time() < deadline:
            # Generate offspring
            offspring = []
            offspring_f = []
            
            for i in range(lam):
                if time.time() > deadline:
                    break
                # Sample
                z = rng.standard_normal(dim)
                y = x_mean + sigma * (C_sqrt @ z)
                # Clip to box? The task says func clips, but we should probably clip 
                # to keep the internal state consistent if we were doing gradient based.
                # CMA-ES usually doesn't clip, but here the box is hard.
                # The prompt says "func(x) clips a finite length-5 vector to the box".
                # So we can pass unclipped y, and func handles it.
                # However, for the mean update, we should probably stay within bounds
                # or let the mean drift. Standard CMA-ES lets the mean drift.
                
                fy = func(y)
                calls_made += 1
                offspring.append(y)
                offspring_f.append(fy)
            
            if len(offspring) == 0:
                break
            
            # Sort offspring
            indices = np.argsort(offspring_f)
            offspring = [offspring[i] for i in indices]
            offspring_f = [offspring_f[i] for i in indices]
            
            # Update mean
            x_new_mean = np.sum([weights[i] * offspring[i] for i in range(mu)], axis=0)
            
            # Update sigma (CSA)
            y_sigma = (x_new_mean - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_sigma
            
            # Update C (CMS)
            y_c = (x_new_mean - x_mean) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_c
            
            h_sigma = 1.0 if np.sqrt(1 - (1 - c_sigma)**2 * (calls_made - 1)) / np.sqrt(1 - (1 - c_sigma)**2 * (calls_made)) < (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (dim + 1)) else 0.0
            # Simplified h_sigma check
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2 * (calls_made - 1)) < (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (dim + 1)) * np.sqrt(1 - (1 - c_sigma)**2 * (calls_made)):
                 h_sigma = 1.0
            else:
                 h_sigma = 0.0
            
            # Rank-one update
            rank_one = np.outer(p_c, p_c)
            h_delta = (1 - h_sigma) * c_c * (2 - c_c) * np.eye(dim)
            
            # Rank-mu update
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (offspring[i] - x_mean) / sigma
                rank_mu += weights[i] * np.outer(y_i, y_i)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (rank_one + h_delta) + c_mu * rank_mu
            
            # Regularization for C
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2.0
            # Simple regularization: add epsilon to diagonal if needed, or recompute
            # For dim=5, we can just recompute Cholesky. If it fails, we have a problem.
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Fallback: reset C to identity or add identity
                C = C + 1e-6 * np.eye(dim)
                try:
                    L = np.linalg.cholesky(C)
                except:
                    C = np.eye(dim)
                    L = np.eye(dim)
            
            C_sqrt = L
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
            
            # Update mean
            x_mean = x_new_mean
            
            # Track best
            if offspring_f[0] < best_f:
                best_f = offspring_f[0]
                best_x = offspring[0].copy()
                
            # Termination criteria
            if sigma < 1e-10:
                break
            if np.linalg.norm(x_mean) > 100: # Safety
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
