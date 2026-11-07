"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer for 5D Rastrigin."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s deadline with 50s safety margin
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        calls = [0]
        
        def call_func(x):
            calls[0] += 1
            return func(x)
        
        # Phase 1: Coarse random exploration (15 calls)
        best_x = None
        best_f = np.inf
        for i in range(15):
            if calls[0] >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = call_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        # Phase 2: CMA-ES (main optimization)
        n = dim
        lambda_ = 4 + int(3 * np.log(n))  # population size
        mu = lambda_ // 2  # number of recombination points
        
        # CMA-ES parameters
        cs = (4 + n / 3) / (n + 4)
        cmu = 2 / ((n + 1) ** 2 + 1)
        c1 = (2 + 2 / (n + 1)) / ((n + 2) ** 2 + 2)
        cc = (4 + n / n) / (n + 4)
        
        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1 / np.sum(w ** 2)
        
        # Initial state
        x_mean = best_x.copy()
        sigma = 1.0
        C = np.eye(n)
        C_eigenvalues = np.ones(n)
        ps = np.zeros(n)
        pc = np.zeros(n)
        
        # Precompute sqrt of C (initially identity)
        def update_cma_ess():
            nonlocal C, C_eigenvalues
            eigvals, eigvecs = np.linalg.eigh(C)
            C_eigenvalues = np.clip(eigvals, 1e-20, None)
            return eigvals, eigvecs
        
        # CMA-ES main loop
        while calls[0] < budget and time.time() < deadline:
            # Generate population
            offspring = []
            for i in range(lambda_):
                if calls[0] >= budget or time.time() > deadline:
                    break
                # Generate random direction
                z = rng.standard_normal(n)
                # Transform using Cholesky-like update
                # For simplicity, use the fact that C starts as identity
                # and update it properly
                y = sigma * z
                xi = x_mean + y
                xi = np.clip(xi, -5, 5)
                fi = call_func(xi)
                offspring.append((fi, xi, z))
            
            if len(offspring) < lambda_:
                break
            
            # Sort by fitness
            offspring.sort(key=lambda t: t[0])
            
            # Update best
            if offspring[0][0] < best_f:
                best_f = offspring[0][0]
                best_x = offspring[0][1].copy()
            
            # Recombination
            x_new = np.zeros(n)
            for i in range(mu):
                x_new += w[i] * offspring[i][1]
            
            # Update step size sigma
            # Success rule: compare distance of new mean to old mean
            diff = x_new - x_mean
            # Approximate 1/sqrt(C) norm
            eigvals, eigvecs = np.linalg.eigh(C)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(np.clip(eigvals, 1e-20, None))) @ eigvecs.T
            norm_diff = np.linalg.norm(diff @ C_inv_sqrt)
            sigma *= np.exp(cs * (norm_diff / sigma - 1) / (n + 2))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update evolution path
            eigvals, eigvecs = np.linalg.eigh(C)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(np.clip(eigvals, 1e-20, None))) @ eigvecs.T
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (C_inv_sqrt @ diff) / sigma
            
            # Update covariance matrix C
            # Rank-1 update
            C += c1 * (np.outer(ps, ps) - C)
            # Rank-mu update
            if mu > 1:
                yw = np.array([offspring[i][1] - x_mean for i in range(mu)])
                C += cmu * (np.outer(yw.T, yw) / (sigma ** 2) - C)
            
            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-12, None)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            x_mean = x_new
        
        # Phase 3: Fine-tuning with smaller steps
        if calls[0] < budget:
            sigma = 0.05
            x = best_x.copy()
            fx = best_f
            for i in range(budget - calls[0]):
                if calls[0] >= budget or time.time() > deadline:
                    break
                step = sigma * rng.standard_normal(dim)
                y = np.clip(x + step, -5, 5)
                fy = call_func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.02
                else:
                    sigma *= 0.95
                if sigma < 0.001:
                    break
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        
        return best_x, best_f
# EVOLVE-BLOCK-END
