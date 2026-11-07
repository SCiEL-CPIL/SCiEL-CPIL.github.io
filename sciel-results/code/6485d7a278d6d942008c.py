import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        # Keep a safety margin to avoid timeout. 330s limit, use 300s.
        deadline = start_time + 295.0 

        # CMA-ES Parameters
        # For dim=5, lambda=15 is standard and effective.
        # 500 calls / 15 = 33 generations.
        lambda_ = 15
        mu = lambda_ // 2  # Number of selected individuals
        
        # Initial state
        # Start at origin, which is within the box and often close to the shifted minimum
        x_mean = np.zeros(dim)
        sigma = 2.0  # Initial step size, large enough to explore [-5,5]
        C = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Adaptation parameters
        c_sigma = (mu + 2) / (dim + mu + 5)
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        d_sigma = 1 + 2 * max(0, (math.sqrt(dim) - 1) / (dim + 3) - 1) + c_sigma
        c_cov = (2 + 2 * mu / dim) / (dim + 1 + 2 * mu / dim)
        
        # Precompute weights for mean update
        # Standard CMA-ES weights
        weights = np.array([np.log(mu/2 + 1) - np.log(i+1) for i in range(mu)])
        weights = weights / np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)
        
        # E||N(0, I)|| for sigma adaptation
        e_norm = math.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        
        # Initial evaluation
        best_x = x_mean.copy()
        best_f = float('inf')
        calls_used = 0
        
        # Helper to evaluate and track best
        def eval_func(x):
            nonlocal best_x, best_f, calls_used
            if calls_used >= budget:
                return best_f
            if time.time() > deadline:
                return best_f
            try:
                f_val = func(x)
            except Exception:
                return best_f
            calls_used += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val
        
        # Initial point evaluation
        best_f = eval_func(x_mean)
        if calls_used >= budget or time.time() > deadline:
            return best_x, best_f

        while calls_used < budget:
            # Check time
            if time.time() > deadline:
                break
                
            # Determine how many candidates we can evaluate
            n_evals = min(lambda_, budget - calls_used)
            
            # Sample candidates
            # Cholesky decomposition of C
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                L = np.eye(dim)
            
            candidates = []
            fvals = []
            
            for i in range(n_evals):
                # Standard normal
                z = np.random.randn(dim)
                # Transform
                x_new = x_mean + sigma * (L @ z)
                # Clip to bounds [-5, 5]
                x_new = np.clip(x_new, -5, 5)
                
                # Evaluate
                f_new = eval_func(x_new)
                candidates.append(x_new)
                fvals.append(f_new)
                
                if calls_used >= budget or time.time() > deadline:
                    break
            
            if len(candidates) < mu:
                # Not enough evaluations to update properly, break
                break
                
            # Sort candidates by fitness
            sorted_indices = np.argsort(fvals[:len(candidates)])
            
            # Update mean
            x_new_mean = np.zeros(dim)
            for i in range(mu):
                idx = sorted_indices[i]
                x_new_mean += weights[i] * candidates[idx]
            
            # Update covariance
            # y_i = (x_i - x_mean) / sigma
            y_vecs = [(candidates[sorted_indices[i]] - x_mean) / sigma for i in range(mu)]
            
            # Rank-one update term
            rank_one = np.zeros((dim, dim))
            for i in range(mu):
                rank_one += weights[i] * np.outer(y_vecs[i], y_vecs[i])
            
            # Update p_sigma
            # p_sigma = (1 - c_sigma) * p_sigma + c_sigma * sqrt(mu_eff) * (x_new_mean - x_mean) / (sigma * ||C^{1/2}||_F)
            # Approximation for ||C^{1/2}||_F is sqrt(trace(C))
            c_norm = np.sqrt(np.trace(C))
            if c_norm < 1e-12:
                c_norm = 1.0
            
            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * np.sqrt(mu_eff) * (x_new_mean - x_mean) / (sigma * c_norm)
            
            # Update sigma
            # sigma = sigma * exp(c_sigma / d_sigma * (||p_sigma|| / E||N(0, I)|| - 1))
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / e_norm - 1))
            
            # Update p_c
            # p_c = (1 - c_c) * p_c + c_c * sqrt(mu_eff) * (x_new_mean - x_mean) / sigma
            p_c = (1 - c_c) * p_c + c_c * np.sqrt(mu_eff) * (x_new_mean - x_mean) / sigma
            
            # Update C
            # C = (1 - c_cov * (1 + (1 - mu_eff) * c_sigma * (1 - c_sigma))) * C 
            #    + c_cov * (mu_eff * p_sigma * p_sigma.T + rank_one)
            c_cov_eff = c_cov * (1 + (1 - mu_eff) * c_sigma * (1 - c_sigma))
            C = (1 - c_cov_eff) * C + c_cov_eff * (mu_eff * np.outer(p_sigma, p_sigma) + rank_one)
            
            # Symmetrize C
            C = (C + C.T) / 2
            
            # Update mean
            x_mean = x_new_mean
            # Clip mean to bounds to keep it in the feasible region
            x_mean = np.clip(x_mean, -5, 5)
            
            # If sigma becomes extremely small, we might stagnate, but budget is limited
            if sigma < 1e-12:
                break

        return best_x, best_f
