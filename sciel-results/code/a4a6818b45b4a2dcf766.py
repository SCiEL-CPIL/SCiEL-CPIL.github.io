import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 330s kill
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        def eval_func(x):
            nonlocal calls_used
            if calls_used >= self.budget:
                return np.inf
            calls_used += 1
            return func(x)
        
        # CMA-ES implementation
        dim = self.dim
        budget = self.budget
        
        # Population size
        lam = 4 + int(3 * np.log(dim))  # ~13 for dim=5
        # We'll use a smaller population to conserve budget
        lam = min(lam, 12)
        
        # Initial parameters
        x0 = np.zeros(dim)
        sigma = 2.0  # Initial step size (Rastrigin has many local minima)
        
        # CMA-ES parameters
        mu = lam // 2  # Number of selected individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / weights.sum()
        
        # Adaptation rates
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * ((mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff)))
        
        # Covariance matrix
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        
        # Evolution path
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Helper for matrix square root (eigen decomposition)
        def mat_sqrt(M):
            eigvals, eigvecs = np.linalg.eigh(M)
            eigvals = np.maximum(eigvals, 1e-12)
            return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T
        
        # Track best
        if time.time() > deadline:
            return None, np.inf
        
        # Initial evaluation
        fx0 = eval_func(x0)
        best_f = fx0
        best_x = x0.copy()
        
        # Main CMA-ES loop
        while calls_used < budget:
            if time.time() > deadline:
                break
            
            # Generate population
            ys = []
            for i in range(lam):
                if calls_used + 1 > budget or time.time() > deadline:
                    break
                zi = rng.standard_normal(dim)
                yi = x0 + sigma * (C_sqrt @ zi)
                yi = np.clip(yi, -5, 5)
                yi = yi.copy()
                ys.append(yi)
            
            if len(ys) == 0:
                break
            
            # Evaluate population
            fvals = []
            for y in ys:
                if calls_used >= budget or time.time() > deadline:
                    break
                f = eval_func(y)
                fvals.append(f)
            
            if len(fvals) < 2:
                break
            
            # Sort by fitness
            order = np.argsort(fvals)
            ys_sorted = [ys[i] for i in order]
            fvals_sorted = [fvals[i] for i in order]
            
            # Update best
            if fvals_sorted[0] < best_f:
                best_f = fvals_sorted[0]
                best_x = ys_sorted[0].copy()
            
            # Update mean (recombination)
            x_mean_old = x0.copy()
            x0 = sum(weights[i] * ys_sorted[i] for i in range(mu))
            
            # Update evolution paths
            # First, compute C^{-1/2}
            C_inv_sqrt = mat_sqrt(np.linalg.inv(C))
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * (C_inv_sqrt @ (x0 - x_mean_old) / sigma)
            
            # Update p_c
            h_sigma = (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (1)))) < (1 + 2 / (dim + 1)) * np.sqrt(c_sigma * (2 - c_sigma))
            h_sigma = 1 if h_sigma else 0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (x0 - x_mean_old) / sigma
            
            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            rank_one = np.outer(p_c, p_c)
            rank_mu = sum(weights[i] * np.outer(ys_sorted[i] - x_mean_old, ys_sorted[i] - x_mean_old) for i in range(mu))
            
            C = (1 + c_1 * delta_h - c_1 - c_mu) * C + c_1 * (rank_one + delta_h * C) + c_mu * rank_mu
            
            # Ensure C is positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            
            # Update C_sqrt
            C_sqrt = mat_sqrt(C)
            
            # Update sigma
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            
            # Convergence check
            if sigma < 1e-8:
                break
            if np.all(np.abs(C - np.eye(dim) * np.trace(C) / dim) < 1e-10):
                break
        
        # Final polish with small perturbations if budget remains
        if best_x is not None and calls_used < budget:
            x = best_x.copy()
            fx = best_f
            sigma_polish = 0.1
            for _ in range(min(20, budget - calls_used)):
                if time.time() > deadline:
                    break
                y = x + sigma_polish * rng.standard_normal(dim)
                y = np.clip(y, -5, 5)
                fy = eval_func(y)
                if fy < fx:
                    x, fx = y, fy
                    best_f = fx
                    best_x = x.copy()
                    sigma_polish *= 1.1
                else:
                    sigma_polish *= 0.85
        
        return best_x, best_f
