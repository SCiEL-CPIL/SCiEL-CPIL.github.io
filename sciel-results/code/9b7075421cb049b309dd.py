"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        rng = np.random.default_rng(12345)

        calls = 0
        best_x = None
        best_f = np.inf

        def count(x):
            nonlocal calls
            fx = func(x)
            calls += 1
            return fx

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))  # 8 for dim=5
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        sigma_0 = 1.0
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim**2))

        def run_cmaes(x0, f0, max_calls):
            nonlocal calls, best_x, best_f
            
            # Initialize
            x_m = x0.copy()
            sigma = sigma_0
            B = np.eye(dim)
            D = 1.0
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            C = np.eye(dim)
            
            # Pre-compute some constants
            sqrt_c1 = np.sqrt(c_1)
            sqrt_cmu = np.sqrt(c_mu)
            
            # We don't reuse f0 in the population to keep CMA-ES standard, 
            # but we have already evaluated x0. 
            # To save budget, we could inject it, but standard CMA-ES is safer.
            
            for gen in range(int(max_calls // lambda_)):
                if calls >= budget or time.time() > deadline:
                    break
                
                # Sample population
                ys = rng.standard_normal((lambda_, dim))
                xs = x_m + sigma * (ys @ (B * D).T)
                xs = np.clip(xs, -5, 5)
                
                fs = np.array([count(x) for x in xs])
                
                # Update best
                idx_best = np.argmin(fs)
                if fs[idx_best] < best_f:
                    best_f = fs[idx_best]
                    best_x = xs[idx_best].copy()
                
                # Sort
                order = np.argsort(fs)
                ys = ys[order]
                fs = fs[order]
                xs = xs[order]
                
                # Weighted mean
                x_m_new = np.sum(weights[:, None] * xs[:mu], axis=0)
                
                # Step size control
                y_m = (x_m_new - x_m) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_m
                h_sigma = 1 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**2) / chi_n) < (1.4 + 2 / (dim + 1)) else 0
                
                # Covariance matrix adaptation
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * y_m
                
                delta_h = (h_sigma - 1) * c_c * (2 - c_c)
                
                # Rank-mu update
                rank_mu = np.zeros((dim, dim))
                for i in range(mu):
                    rank_mu += weights[i] * np.outer(ys[i], ys[i])
                    
                C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * np.outer(p_c, p_c) + c_mu * rank_mu
                
                # Ensure symmetry
                C = (C + C.T) / 2
                
                # Eigen decomposition for B and D
                eigvals, eigvecs = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(eigvals, 1e-12))
                B = eigvecs
                
                x_m = x_m_new
                
                # Sigma update
                sigma *= np.exp(c_sigma / chi_n * (np.linalg.norm(p_sigma) / chi_n - 1))
                
                # Check convergence
                if np.max(D) / np.min(D) > 1e14 or sigma < 1e-15:
                    break

        # Phase 1: Multi-start CMA-ES
        n_starts = 3
        calls_per_start = (budget - 50) // n_starts  # Reserve some for potential polish or just distribute
        
        for s in range(n_starts):
            if calls >= budget or time.time() > deadline:
                break
            
            # Diverse starting points
            if s == 0:
                x0 = np.zeros(dim)
            elif s == 1:
                x0 = rng.uniform(-4, 4, dim)
            else:
                x0 = rng.uniform(-5, 5, dim)
            
            x0 = np.clip(x0, -5, 5)
            f0 = count(x0)
            
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
            
            # Run CMA-ES
            run_cmaes(x0, f0, calls_per_start)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
