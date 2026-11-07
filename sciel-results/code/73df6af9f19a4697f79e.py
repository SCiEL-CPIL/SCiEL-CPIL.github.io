import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # CMA-ES parameters
        # Population size
        lam = 4 + int(3 * np.log(dim))
        # Number of recombination weights
        mu = lam // 2
        # Recombination weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights ** 2)
        
        # Learning rates
        c_sigma = 1 / (mu_eff + 3)
        c_c = (4 + mu_eff / lam) / ((lam + 4) * (1 + 2 * mu_eff / lam))
        c_1 = 2 / ((lam + 1.3) ** 2 + mu_eff)
        c_mu = 2 * (mu_eff - 2 + 1 / mu_eff) / ((lam + 2) ** 2 + mu_eff)
        
        # Initial state
        x_mean = np.zeros(dim)
        sigma = 2.0
        C = np.eye(dim)
        B = np.eye(dim)
        D = np.ones(dim)
        
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Track best
        best_x = x_mean.copy()
        best_f = np.inf
        calls_made = 0
        start_time = time.time()
        
        # Helper to evaluate and update best
        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Initial evaluation of the mean (optional but good for baseline)
        # Actually, we usually start with the population.
        
        while calls_made < budget:
            # Check time
            if time.time() - start_time > 290:  # Safety margin
                break
                
            # Sample population
            X = np.zeros((lam, dim))
            for i in range(lam):
                z = np.random.randn(dim)
                # Transform z using B and D
                # C = B D^2 B^T
                # y = B D z
                y = B @ (D * z)
                X[i] = x_mean + sigma * y
            
            # Evaluate population
            F = np.zeros(lam)
            for i in range(lam):
                if calls_made >= budget:
                    break
                # Clip to box [-5, 5]
                x_clip = np.clip(X[i], -5, 5)
                F[i] = evaluate(x_clip)
            
            if calls_made >= budget:
                break
                
            # Sort by function value
            idx = np.argsort(F)
            X = X[idx]
            F = F[idx]
            
            # Recombination
            x_new = np.zeros(dim)
            for i in range(mu):
                x_new += weights[i] * X[i]
            
            # Update sigma (step size)
            # h_sigma check
            # norm of B^T D^-1 p_sigma
            norm_Bt_D_inv_p_sigma = np.linalg.norm(B.T @ (p_sigma / D))
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
            h_sigma = 1 if norm_Bt_D_inv_p_sigma < (1.4 + 2 / (dim + 1)) * chi_n else 0
            
            p_sigma = (1 - c_sigma) * p_sigma + h_sigma * np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B.T @ (D * ((x_new - x_mean) / sigma)))
            
            sigma *= np.exp(c_sigma / c_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            # Prevent sigma from becoming too small or too large
            sigma = np.clip(sigma, 1e-10, 1e6)
            
            # Update C (covariance matrix)
            # Rank-mu update
            v = (x_new - x_mean) / sigma
            p_c = (1 - c_c) * p_c + h_sigma * c_c * v
            
            # h_c check
            norm_Bt_D_inv_p_c = np.linalg.norm(B.T @ (p_c / D))
            norm_Bt_D_inv_p_sigma_2 = np.linalg.norm(B.T @ (p_sigma / D)) ** 2
            h_c = 1 if norm_Bt_D_inv_p_c * (norm_Bt_D_inv_p_c - (1 - c_c)) < (2 + 2 / c_c) * norm_Bt_D_inv_p_sigma_2 else 0
            
            C += (c_1 + c_mu) * (np.outer(p_c, p_c) / (1 - (1 - c_c) ** 2) - C)
            for i in range(mu):
                v_i = (X[i] - x_mean) / sigma
                C += c_mu * weights[i] * np.outer(v_i, v_i)
            
            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            # Regularization
            C += 1e-12 * np.eye(dim)
            
            # Update B and D via eigendecomposition
            D, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 1e-12))
            
            # Update mean
            x_mean = x_new
            
            # Check convergence
            if np.linalg.norm(x_new - x_mean) < 1e-6:
                # Simple restart or break if stuck
                # For now, just break if sigma is very small
                if sigma < 1e-4:
                    break

        return best_x, best_f
