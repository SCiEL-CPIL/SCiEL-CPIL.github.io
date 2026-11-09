import numpy as np
import time
import itertools
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin
        
        n = self.dim
        budget = self.budget
        
        # Phase 1: Deterministic Lattice Search (3^5 = 243 points)
        grid_vals = [-5.0, 0.0, 5.0]
        grid_points = [np.array(combo) for combo in itertools.product(grid_vals, repeat=n)]
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        X_obs = []
        Y_obs = []
        
        # Evaluate grid points
        for x in grid_points:
            if calls_used >= budget or time.time() > deadline:
                break
            fx = func(x)
            calls_used += 1
            X_obs.append(x)
            Y_obs.append(fx)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        if best_x is None:
            # Fallback
            best_x = np.zeros(n)
            best_f = func(best_x)
            calls_used += 1
            X_obs.append(best_x)
            Y_obs.append(best_f)
            
        # Phase 2: Bayesian Optimization with GP
        remaining_budget = budget - calls_used
        
        if remaining_budget <= 0:
            return best_x, best_f
            
        # GP Hyperparameters
        # Weierstrass is oscillatory. Length scale 1.0 is a guess.
        # Signal variance estimated from data.
        length_scale = 1.0
        
        # Acquisition function grid
        # 8^5 = 32768 points
        acq_grid_vals = np.linspace(-5, 5, 8)
        acq_grid = np.array(list(itertools.product(acq_grid_vals, repeat=n)))
        
        # (1+1)-ES as fallback if BO fails or takes too long? No, stick to BO.
        
        while calls_used < budget:
            if time.time() > deadline:
                break
            
            X = np.array(X_obs)
            Y = np.array(Y_obs)
            
            # Estimate signal variance
            sigma_f_sq = np.var(Y) + 1e-8
            if np.isnan(sigma_f_sq) or sigma_f_sq < 1e-8:
                sigma_f_sq = 1.0
                
            # Compute Kernel matrix K (N x N)
            # Matern 5/2
            diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=-1))
            r = dists / length_scale
            K = sigma_f_sq * (1 + np.sqrt(5)*r + (5/3)*r**2) * np.exp(-np.sqrt(5)*r)
            
            # Add small jitter for numerical stability
            K += 1e-6 * np.eye(len(X))
            
            # Solve K alpha = Y
            try:
                alpha = np.linalg.solve(K, Y)
            except np.linalg.LinAlgError:
                alpha = np.linalg.lstsq(K, Y, rcond=None)[0]
                
            # Compute EI on acquisition grid
            # k_x = K(x, X) (M x N)
            diff_x = acq_grid[:, np.newaxis, :] - X[np.newaxis, :, :]
            dists_x = np.sqrt(np.sum(diff_x**2, axis=-1))
            r_x = dists_x / length_scale
            k_x = sigma_f_sq * (1 + np.sqrt(5)*r_x + (5/3)*r_x**2) * np.exp(-np.sqrt(5)*r_x)
            
            # mu = k_x @ alpha
            mu = k_x @ alpha
            
            # sigma^2 = k(x,x) - k_x @ K^-1 @ k_x^T
            # k(x,x) is sigma_f_sq
            # We need K^-1. We can compute it or use alpha trick?
            # sigma^2 = sigma_f_sq - k_x @ np.linalg.solve(K, k_x.T)
            # This requires solving K Z = k_x.T for each x? No, Z is N x M.
            # K Z = k_x.T
            try:
                Z = np.linalg.solve(K, k_x.T)
            except np.linalg.LinAlgError:
                Z = np.linalg.lstsq(K, k_x.T, rcond=None)[0]
                
            var_x = sigma_f_sq - np.sum(k_x * Z, axis=1)
            # Clip negative variance
            var_x = np.maximum(var_x, 1e-12)
            sigma_x = np.sqrt(var_x)
            
            # Expected Improvement
            f_min = np.min(Y)
            Z_ei = (f_min - mu) / sigma_x
            # Handle division by zero
            Z_ei = np.where(sigma_x > 1e-8, Z_ei, 0)
            
            ei = (f_min - mu) * norm.cdf(Z_ei) + sigma_x * norm.pdf(Z_ei)
            
            # Find best EI point
            idx = np.argmax(ei)
            x_new = acq_grid[idx]
            
            # Evaluate
            f_new = func(x_new)
            calls_used += 1
            
            # Update
            X_obs.append(x_new)
            Y_obs.append(f_new)
            
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
                
        return best_x, best_f
