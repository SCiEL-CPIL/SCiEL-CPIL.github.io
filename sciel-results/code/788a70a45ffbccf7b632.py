"""Rewritten optimizer using CMA-ES for BBOB-style f18."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer for minimizing func on [-5, 5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 300  # Safety margin for 330s limit
        
        # CMA-ES parameters
        lam = 10  # Population size
        mu = lam // 2  # Number of weights
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Strategy parameters
        cc = (4 + mueff / d) / (d + 4 + 2 * mueff / d)
        cs = (mueff + 2) / (d + mueff + 5)
        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (d + 1)) - 1) + cs
        cmu = min(1 - cc, 2 * ((mueff + 1) / (d + 2))**2 - 1)
        chiN = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
        
        # Initialize state
        m = np.zeros(d)
        sigma = 2.0
        C = np.eye(d)
        B = np.eye(d)
        D = np.ones(d)
        
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        calls_made = 0
        best_x = m.copy()
        best_f = func(best_x)
        calls_made += 1
        
        while calls_made < budget:
            if time.time() > deadline - 5:
                break
                
            # Check if we have enough budget for a full generation
            if budget - calls_made < lam:
                # Do individual evaluations if not enough for a full generation
                # This is a fallback, but with 500 budget and lam=10, we are fine.
                # We just stop if we can't do a full generation to maintain CMA-ES integrity.
                # Or we can do a final polish with (1+1)-ES or similar.
                # Let's just break.
                break

            # Generate population
            Z = np.random.randn(lam, d)
            Y = (B @ np.diag(np.sqrt(D)) @ Z.T).T
            X = m + sigma * Y
            X = np.clip(X, -5, 5)
            
            # Evaluate
            F = np.array([func(x) for x in X])
            calls_made += lam
            
            # Update best
            min_idx = np.argmin(F)
            if F[min_idx] < best_f:
                best_f = F[min_idx]
                best_x = X[min_idx].copy()
            
            # Sort
            order = np.argsort(F)
            X = X[order]
            F = F[order]
            Y = Y[order]
            
            # Update mean
            m_new = m + sigma * np.sum(weights[:, np.newaxis] * X[:mu], axis=0)
            
            # Update p_sigma
            diff = (m_new - m) / sigma
            btd = B.T @ diff
            c_inv_sqrt_diff = B @ (btd / np.sqrt(D))
            
            p_sigma = (1 - cs) * p_sigma + np.sqrt(cs * (2 - cs) * mueff) * c_inv_sqrt_diff
            
            # Update p_c
            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sigma = 1.0 if norm_p_sigma < 1.4 * chiN else 0.0
            
            p_c = (1 - cc) * p_c + h_sigma * np.sqrt(cc * (2 - cc) * mueff) * c_inv_sqrt_diff
            
            # Update C
            rank_mu_term = np.sum(weights[:, np.newaxis, np.newaxis] * Y[:mu, :, np.newaxis] * Y[:mu, np.newaxis, :], axis=0)
            
            alpha_c = 1 - (mueff + 2 * cmu) * cs * (2 - cs) + cmu
            C = alpha_c * C + cmu * rank_mu_term + (h_sigma - 1) * cc * (2 - cc) * np.outer(p_c, p_c)
            
            C = (C + C.T) / 2
            
            # Update B and D
            try:
                D, B = np.linalg.eigh(C)
                D = np.maximum(D, 1e-12)
            except np.linalg.LinAlgError:
                C = np.eye(d)
                D = np.ones(d)
                B = np.eye(d)
            
            # Update sigma
            sigma = sigma * np.exp(cs / damps * (np.linalg.norm(p_sigma) / chiN - 1))
            
            # Update m
            m = m_new
            
            if sigma < 1e-10:
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
