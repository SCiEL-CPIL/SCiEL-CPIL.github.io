import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Deadline guard: 300s limit, use 290s to be safe
        deadline = time.time() + 290
        
        dim = self.dim
        budget = self.budget
        
        # CMA-ES Parameters
        # Population size lambda
        lam = 4 + int(3 * np.log(dim))
        # Number of parents
        mu = lam // 2
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        # Effective population size
        mueff = 1.0 / np.sum(weights**2)
        
        # Adaptation rates
        cc = (4.0 + mueff / dim) / (dim + 4.0 + 2.0 * mueff / dim)
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        ds = 1.0 + 2.0 * max(0.0, np.sqrt((mueff - 1.0) / (dim + 1.0)) - 1.0) + cs
        c1 = (2.0 + 2.0 * mueff / (dim + 1.0)) / ((dim + 1.3)**2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0)**2 + mueff))
        
        # State
        x_mean = np.zeros(dim) # Start at origin, good for Katsuura
        sigma = 2.0
        C = np.eye(dim)
        Cinv = np.eye(dim)
        Cs = np.zeros(dim)
        hs = 0.0
        best_x = x_mean.copy()
        best_f = func(best_x)
        
        calls_made = 1
        rng = np.random.default_rng(42)
        
        # Precompute sqrt(C) for sampling? No, C changes.
        # We need to sample from N(0, C).
        
        while calls_made < budget and time.time() < deadline:
            # Check if time is running out to finish a generation
            # If we have less than 5 seconds, break to ensure we return
            if time.time() > deadline - 5:
                break
                
            # Generate samples
            # Sample from N(0, I) and transform
            z = rng.standard_normal((lam, dim))
            
            # Transform to N(0, C)
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If C is not positive definite, reset
                C = np.eye(dim)
                Cinv = np.eye(dim)
                L = np.eye(dim)
            
            samples = x_mean + sigma * (z @ L.T)
            
            # Clip to bounds
            samples = np.clip(samples, -5.0, 5.0)
            
            # Evaluate
            fitness = np.array([func(s) for s in samples])
            calls_made += lam
            
            # Sort
            idx = np.argsort(fitness)
            fitness = fitness[idx]
            samples = samples[idx]
            
            # Update best
            if fitness[0] < best_f:
                best_f = fitness[0]
                best_x = samples[0]
                
            # Update mean
            w = weights
            x_new = np.sum(samples[:mu] * w[:, np.newaxis], axis=0)
            
            # Cumulative step
            p_sigma = (1.0 - cs) * p_sigma + np.sqrt(cs * (2.0 - cs) * mueff) * (np.linalg.cholesky(C).T @ (x_new - x_mean) / sigma)
            
            # Note: The standard CMA-ES update for p_sigma uses C^{-1/2}
            # Let's use the standard formula:
            # p_sigma = (1-cs)*p_sigma + sqrt(cs*(2-cs)*mueff) * C^{-1/2} (x_new - x_mean)/sigma
            
            # We need C^{-1/2}. We have L such that C = L L^T. So C^{-1/2} = L^{-T} L^{-1}? 
            # Actually, sampling was z @ L.T, so x = L.T z. C = L.T L. 
            # Wait, if x = L.T z, Cov(x) = L.T I L = L.T L.
            # So C = L.T L.
            # C^{-1/2} is not simply related to L unless we are careful.
            # Standard implementation often uses eigenvalue decomposition or just solves.
            # For simplicity and speed, let's use the update formula that avoids explicit C^{-1/2} if possible,
            # or compute it.
            
            # Let's stick to a slightly simplified but robust CMA-ES implementation.
            # Actually, the previous line `p_sigma = ...` was a placeholder.
            # Let's do it properly.
            
            # Re-evaluate p_sigma update
            # We need C^{-1/2}. 
            # Since C = L.T L, C^{-1/2} is not L^{-T}. 
            # Let's use the fact that we sampled y = L.T z. 
            # The step in "C-space" is z.
            # x_new - x_mean = sigma * L.T * (sum w_i z_i)
            # So C^{-1/2} (x_new - x_mean) / sigma = C^{-1/2} L.T (sum w_i z_i)
            # This is getting complicated. Let's use the standard Numpy-based CMA-ES logic.
            
            # Reset p_sigma logic to standard:
            # p_sigma = (1-cs)*p_sigma + sqrt(cs*(2-cs)*mueff) * C^{-1/2} (x_new - x_mean)/sigma
            
            # Compute C^{-1/2}
            # Eigen decomposition
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            except:
                C_inv_sqrt = np.eye(dim)
                
            p_sigma = (1.0 - cs) * p_sigma + np.sqrt(cs * (2.0 - cs) * mueff) * (C_inv_sqrt @ (x_new - x_mean) / sigma)
            
            hs = np.linalg.norm(p_sigma)
            hsig = (hs / np.sqrt(1.0 - (1.0 - cs)**(2 * (calls_made // lam + 1)))) < (1.4 + 2.0 / (dim + 1.0)) * np.sqrt(1.0 - (1.0 - cs)**(2 * (calls_made // lam + 1)))
            
            # Update C
            if hsig:
                Cs = (1.0 - cs) * Cs + np.sqrt(cs * (2.0 - cs) * mueff) * (x_new - x_mean) / sigma
            
            # Rank-1 update
            v = (x_new - x_mean) / sigma
            C = (1.0 + c1 * (2.0 - c1) * (hsig - 1.0) + c1 * mueff) * C \
                - c1 * (2.0 - c1) * (hsig - 1.0) * C \
                + c1 * (np.outer(v, v) - (1.0 - hsig) * C)
            
            # Rank-mu update
            # y_i = (x_i - x_mean_old) / sigma
            y = (samples[:mu] - x_mean) / sigma
            C = C + cmu * np.sum(weights[:, np.newaxis, np.newaxis] * np.einsum('ni,nj->nij', y, y), axis=0)
            
            # Ensure symmetry
            C = (C + C.T) / 2.0
            
            # Update sigma
            sigma *= np.exp(cs / ds * (hs / np.sqrt(2.0 * dim) - 1.0))
            
            # Update mean
            x_mean = x_new
            
            # Update Cinv (approx)
            try:
                Cinv = np.linalg.inv(C)
            except:
                Cinv = np.eye(dim)
                
            # Regularize C if eigenvalues become too small
            eigvals, _ = np.linalg.eigh(C)
            if np.min(eigvals) < 1e-12:
                C = C + 1e-12 * np.eye(dim)
                
        return best_x, best_f
