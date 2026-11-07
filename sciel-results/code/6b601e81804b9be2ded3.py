"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES optimizer with deadline guard for 5D box-constrained problems."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # 320s safety margin before 330s limit
        
        rng = np.random.default_rng(12345)
        n = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size: lambda = 4 + 3 * ln(n)
        lam = max(4 + int(3 * np.log(n)), 2 * n + 1)
        lam = min(lam, budget // 2)  # Don't use more than half budget for one generation
        
        # Initial point: center of box
        x = np.zeros(n)
        fx = func(x)
        evals = 1
        
        # Initial sigma
        sigma = 2.0
        
        # Covariance matrix C = I
        C = np.eye(n)
        # Inverse of C
        C_inv = np.eye(n)
        
        # Evolution paths
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        
        # Parameters
        mu = lam // 2  # Number of selected individuals
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 1)) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        
        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))
        
        # Track best
        best_x = x.copy()
        best_f = fx
        
        # Main loop
        while evals < budget:
            # Check deadline
            if time.time() > deadline:
                break
            
            # Generate offspring
            ys = []
            fvals = []
            for i in range(lam):
                if evals >= budget:
                    break
                # Sample from multivariate normal with covariance C
                # y = x + sigma * z where z ~ N(0, C)
                z = rng.multivariate_normal(np.zeros(n), C, size=1).flatten()
                y = x + sigma * z
                # Clip to box
                y = np.clip(y, -5, 5)
                fy = func(y)
                evals += 1
                ys.append(y)
                fvals.append(fy)
                
                if fy < best_f:
                    best_f = fy
                    best_x = y.copy()
            
            if len(ys) == 0:
                break
            
            # Sort by fitness
            order = np.argsort(fvals)
            ys_sorted = [ys[i] for i in order[:mu]]
            fvals_sorted = [fvals[i] for i in order[:mu]]
            
            # Weighted mean of selected
            x_new = np.sum([weights[i] * ys_sorted[i] for i in range(mu)], axis=0)
            
            # Update evolution paths
            # p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv @ (x_new - x) / sigma
            hsig = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (evals + 1))) < (1.4 + 2 / (n + 1)) * chi_n
            hsig = 1 if hsig else 0
            
            # p_c
            p_c = (1 - c_c) * p_c + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv @ (x_new - x) / sigma
            
            # Update covariance matrix
            delta_h = (1 - hsig) * c_c * (2 - c_c)
            C = (1 + c_1 + c_mu - c_1 + delta_h) * C
            C -= c_1 * (np.outer(p_c, p_c) + delta_h * C)
            for i in range(mu):
                y_i = (x_new - x) / sigma
                C += c_mu * weights[i] * np.outer(y_i, y_i)
            
            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Add small regularization if needed
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-10:
                C += 1e-10 * np.eye(n)
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)
            
            # Update mean
            x = x_new
            C_inv = np.linalg.inv(C)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
