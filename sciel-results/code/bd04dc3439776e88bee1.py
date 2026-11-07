"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """CMA-ES implementation for BBOB f11."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0

        # CMA-ES parameters
        lam = 4 + int(3 * np.log(dim))
        if lam < 2:
            lam = 2
        
        mu = lam // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights**2)
        
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2)**2 + mu_eff))
        
        rng = np.random.default_rng(0)
        mean = rng.uniform(-4, 4, dim)
        sigma = 2.0
        C = np.eye(dim)
        
        # Path accumulators
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        best_x = mean.copy()
        best_f = func(best_x)
        calls = 1
        
        # Helper to compute matrix square root and inverse square root
        def update_sqrt():
            nonlocal C
            U, s, Vt = np.linalg.svd(C)
            C_sqrt = U @ np.diag(np.sqrt(s)) @ Vt
            C_inv_sqrt = Vt.T @ np.diag(1 / np.sqrt(s)) @ U.T
            return C_sqrt, C_inv_sqrt

        C_sqrt, C_inv_sqrt = update_sqrt()
        
        while calls < budget:
            if time.time() > deadline:
                break
                
            # Sample
            ys = []
            for _ in range(lam):
                z = rng.standard_normal(dim)
                y = mean + sigma * (C_sqrt @ z)
                y = np.clip(y, -5, 5)
                ys.append(y)
            
            # Evaluate
            fs = np.array([func(y) for y in ys])
            calls += len(ys)
            
            # Sort
            order = np.argsort(fs)
            ys_sorted = [ys[i] for i in order]
            fs_sorted = fs[order]
            
            # Update best
            if fs_sorted[0] < best_f:
                best_f = fs_sorted[0]
                best_x = ys_sorted[0].copy()
                
            # Recombination
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * ys_sorted[i]
                
            # Path update for sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt @ (y_w - mean) / sigma)
            
            # Hessian norm check
            h_sig = 1 if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2 * (calls + 1))) < (1.4 + 2 / (dim + 1)) * np.sqrt(dim + 1 - mu_eff) else 0
            
            # Path update for C
            p_c = (1 - c_c) * p_c + h_sig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (y_w - mean) / sigma
            
            # Update C
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sig) * c_c * (2 - c_c) * C)
            for i in range(mu):
                diff = ys_sorted[i] - mean
                C += c_mu * weights[i] * np.outer(diff, diff)
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(dim) - 1))
            
            # Update mean
            mean = y_w
            mean = np.clip(mean, -5, 5)
            
            # Update square root
            C_sqrt, C_inv_sqrt = update_sqrt()

        return best_x, best_f
# EVOLVE-BLOCK-END
