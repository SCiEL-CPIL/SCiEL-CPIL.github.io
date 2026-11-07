"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES-like optimizer with time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 280s safety margin before 300s limit
        rng = np.random.default_rng(42)
        
        dim = self.dim
        budget = self.budget
        
        # Population size
        lam = 12
        mu = lam // 2
        
        # Initial point: center of box
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        
        # Step size
        sigma = 1.0
        
        # Covariance matrix (start with identity)
        C = np.eye(dim)
        # Inverse square root of C for efficient sampling
        C_eigvals, C_eigvecs = np.linalg.eigh(C)
        C_eigvals = np.clip(C_eigvals, 1e-12, None)
        C_inv_sqrt = (C_eigvecs @ np.diag(1.0 / np.sqrt(C_eigvals))) @ C_eigvecs.T
        
        # Learning rates
        c_sigma = (mu + 0.5) / (dim + 3)
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * (mu - 2 + 1 / mu) / ((dim + 2) ** 2 + mu))
        
        # Path accumulators
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Weighted recombination weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1 / np.sum(w ** 2)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        calls = 1  # initial call
        
        for gen in range(budget // lam):
            # Check time
            if time.time() > deadline:
                break
            
            # Generate population
            offspring = []
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x + sigma * (C_inv_sqrt @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                offspring.append((y, fy))
            
            # Sort by fitness
            offspring.sort(key=lambda t: t[1])
            
            # Update best
            for y, fy in offspring:
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
            
            # Recombination
            x_old = x.copy()
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += w[i] * offspring[i][0]
            
            # Step size control
            z = C_inv_sqrt @ (x_new - x_old) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * z
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (gen + 1))) / chi_n < (1.4 + 2 / (dim + 1))
            if hsig:
                p_c = (1 - c_c) * p_c + c_c * (x_new - x_old) / sigma
            else:
                p_c = (1 - c_c) * p_c
            
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Covariance matrix update
            y = (x_new - x_old) / sigma
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - hsig) * c_c * (2 - c_c) * C) + c_mu * sum(w[i] * np.outer(y - w[i] * p_c, y - w[i] * p_c) for i in range(mu))
            
            # Regularize C
            C_eigvals, C_eigvecs = np.linalg.eigh(C)
            C_eigvals = np.clip(C_eigvals, 1e-12, None)
            C = C_eigvecs @ np.diag(C_eigvals) @ C_eigvecs.T
            C_inv_sqrt = (C_eigvecs @ np.diag(1.0 / np.sqrt(C_eigvals))) @ C_eigvecs.T
            
            x = x_new
            
            # Check if we've used all budget
            if calls >= budget:
                break
        
        return best_x, best_fx
# EVOLVE-BLOCK-END
