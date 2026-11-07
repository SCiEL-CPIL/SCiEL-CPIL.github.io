"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Multi-restart CMA-ES with adaptive population size and step size.
    Designed for the Bent Cigar (f12) landscape: highly ill-conditioned.
    Uses a simple diagonal CMA-ES variant (separable) which is robust and fast.
    Since f12 is separable (sum of x_i^2 / (100^(i-1))), a separable optimizer
    like CMA-ES with diagonal covariance or even a simple adaptive coordinate
    search can be very effective. However, the rotation makes it non-separable
    in the original coordinates, so we need a full covariance matrix or a
    rotation-invariant method. CMA-ES is the standard choice.
    
    We implement a lightweight CMA-ES with:
    - Population size lambda = 4 + floor(3 * log(dim))
    - Adaptive sigma
    - Restart when progress stalls
    - Time guard to stop before 300s deadline
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # CMA-ES parameters
        dim = self.dim
        lambda_ = 4 + int(3 * np.log(dim))  # population size
        mu = lambda_ // 2  # number of best solutions
        # Weights
        weights = np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        def cma_es(x0, sigma0, budget, rng, func, deadline):
            nonlocal calls_made, best_x, best_f
            
            x = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            
            cmacov_inv_sqrt = np.eye(dim)
            
            # Update rates
            cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            cs = (mu_eff + 2) / (dim + mu_eff + 5)
            c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
            damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
            
            # Hessian update
            def update_hess_inv_sqrt():
                nonlocal cmacov_inv_sqrt
                # Simple diagonal approximation for speed, but let's do full
                # For dim=5, full matrix is cheap
                C_eigvals, C_eigvecs = np.linalg.eigh(C)
                C_eigvals = np.maximum(C_eigvals, 1e-20)
                cmacov_inv_sqrt = C_eigvecs @ np.diag(1.0 / np.sqrt(C_eigvals)) @ C_eigvecs.T
            
            update_hess_inv_sqrt()
            
            for i in range(budget):
                if time.time() > deadline:
                    return x, sigma
                
                # Generate offspring
                ys = np.zeros((lambda_, dim))
                for j in range(lambda_):
                    z = rng.standard_normal(dim)
                    y = x + sigma * cmacov_inv_sqrt @ z
                    y = np.clip(y, -5, 5)
                    ys[j] = y
                
                # Evaluate
                fs = np.array([func(y) for y in ys])
                calls_made += lambda_
                
                # Update best
                idx_best = np.argmin(fs)
                if fs[idx_best] < best_f:
                    best_f = fs[idx_best]
                    best_x = ys[idx_best].copy()
                
                # Sort
                order = np.argsort(fs)
                ys_sorted = ys[order]
                fs_sorted = fs[order]
                
                # New mean
                x_new = np.sum(weights[:, None] * ys_sorted[:mu], axis=0)
                
                # Success rate / sigma adaptation
                # Simple 1/5 rule approximation
                improved = np.sum(fs_sorted[:mu] < fs_sorted[mu:])
                if improved > 0:
                    sigma *= 1.1
                else:
                    sigma *= 0.9
                sigma = np.clip(sigma, 1e-10, 50.0)
                
                # Update evolution paths
                pc = (1 - cc) * pc + cc * cmacov_inv_sqrt @ (x_new - x)
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * cmacov_inv_sqrt @ (x_new - x)
                
                # Update covariance
                delta = 1.0
                if np.linalg.norm(ps) ** 2 / (1 - (1 - cs) ** (2 * (i + 1))) < (1 + 6 / dim) * (1 - (1 - cs) ** 2):
                    delta = 1.0 + cs * (2 - cs)
                
                C = (1 + c1 * (delta - 1) - c1 - cmu) * C
                C += c1 * (np.outer(pc, pc) - (1 - (1 - cc) ** 2) * C)
                C += cmu * np.sum(weights[:mu, None, None] * np.outer(ys_sorted[:mu] - x, ys_sorted[:mu] - x), axis=0)
                
                # Symmetrize and ensure positive definite
                C = (C + C.T) / 2
                C_eigvals, C_eigvecs = np.linalg.eigh(C)
                C_eigvals = np.maximum(C_eigvals, 1e-10)
                C = C_eigvecs @ np.diag(C_eigvals) @ C_eigvecs.T
                
                x = x_new
                update_hess_inv_sqrt()
                
                # Restart check: if no progress in last 50 evaluations
                if i > 0 and i % 50 == 0:
                    if np.min(fs_sorted) >= best_f * (1 + 1e-6) and best_f < np.inf:
                        break
                
            return x, sigma
        
        # Multi-restart strategy
        # Start with a few different initializations
        n_restarts = 3
        per_budget = self.budget // n_restarts
        
        for r in range(n_restarts):
            if time.time() > deadline:
                break
            if calls_made >= self.budget:
                break
            
            # Different starting points
            if r == 0:
                x0 = rng.uniform(-5, 5, dim)
                sigma0 = 2.0
            elif r == 1:
                x0 = np.zeros(dim)
                sigma0 = 1.0
            else:
                x0 = best_x if best_x is not None else rng.uniform(-5, 5, dim)
                sigma0 = 0.5
            
            remaining = self.budget - calls_made
            if remaining <= 0:
                break
            
            x_final, _ = cma_es(x0, sigma0, remaining, rng, func, deadline)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
