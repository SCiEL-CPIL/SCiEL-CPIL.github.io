"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with time-aware budget management."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin well before 330s limit
        
        rng = np.random.default_rng(42)
        
        # Initial guess at center
        x = np.zeros(self.dim)
        fx = func(x)
        
        # CMA-ES parameters
        n = self.dim
        lam = 4 + int(3 * np.log(n))  # population size ~ 13
        mu = lam // 2  # number of weights for recombination
        
        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mueff = 1.0 / np.sum(weights**2)
        
        # Step size
        sigma = 2.0
        
        # Covariance matrix
        C = np.eye(n)
        # Eigenvalue decomposition for sqrt(C)
        D, B = np.linalg.eigh(C)
        Csqrt = B @ np.diag(np.sqrt(np.maximum(D, 1e-10))) @ B.T
        Cinv = B @ np.diag(1.0 / np.maximum(D, 1e-10)) @ B.T
        Dinv = 1.0 / np.maximum(D, 1e-10)
        
        # Evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Time constants
        c_sigma = (mueff + 2) / (n + mueff + 5)
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2)**2 + mueff))
        
        # Nudge factors
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n**2))
        
        best_x = x
        best_f = fx
        
        calls_made = 1
        
        while calls_made < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Generate population
            Y = rng.standard_normal((lam, n))
            Y = Y @ Csqrt  # Transform to covariance structure
            X = np.array([x + sigma * y for y in Y])
            
            # Evaluate
            F = []
            for xi in X:
                if time.time() > deadline:
                    break
                fi = func(xi)
                F.append(fi)
                calls_made += 1
            
            if len(F) < lam:
                break
            
            F = np.array(F)
            
            # Sort
            order = np.argsort(F)
            X = X[order]
            F = F[order]
            
            # Update best
            if F[0] < best_f:
                best_f = F[0]
                best_x = X[0].copy()
            
            # Weighted mean
            m = np.sum(weights[:, None] * X[:mu], axis=0)
            
            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (Csqrt.T @ (m - x))
            hsig = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**min(1, calls_made / (n + 1)))) < (1.4 + 2 / (n + 1)) * chi_n else 0
            
            p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c * (2 - c_c) * mueff) * (m - x) / sigma
            
            # Update mean
            x = m
            
            # Update covariance
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            rank_1 = np.outer(p_c, p_c)
            rank_mu = np.sum(weights[:, None, None] * (X[:mu] - m)[:, None, :] @ (X[:mu] - m)[:, :, None], axis=0)
            
            C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C + c_1 * rank_1 + c_mu * rank_mu
            
            # Ensure positive definiteness
            C = (C + C.T) / 2
            D, B = np.linalg.eigh(C)
            D = np.maximum(D, 1e-12)
            C = B @ np.diag(D) @ B.T
            Csqrt = B @ np.diag(np.sqrt(D)) @ B.T
            Cinv = B @ np.diag(1.0 / D) @ B.T
            Dinv = 1.0 / D
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Convergence check
            if np.linalg.norm(m - x) < 1e-10 and sigma < 1e-10:
                break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
