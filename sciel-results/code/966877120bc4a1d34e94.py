import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        deadline = time.time() + 280
        dim = self.dim
        budget = self.budget
        
        calls = 0
        best_x = None
        best_fx = np.inf
        X = []
        Y = []
        
        # GP Hyperparameters
        # Bent Cigar has scale ~1-5. 
        l = 1.5 
        sigma = 1.0
        noise = 1e-6

        def rbf_kernel(X1, X2):
            X1_sq = np.sum(X1**2, axis=1)
            X2_sq = np.sum(X2**2, axis=1)
            dist_sq = X1_sq[:, None] + X2_sq[None, :] - 2 * X1 @ X2.T
            dist_sq = np.maximum(dist_sq, 0)
            return sigma**2 * np.exp(-dist_sq / (2 * l**2))

        def wrapped_func(x):
            nonlocal calls, best_x, best_fx, X, Y
            if time.time() > deadline:
                raise Exception("Time limit exceeded")
            
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = func(x_clipped)
            calls += 1
            
            if fx < best_fx:
                best_fx = fx
                best_x = x_clipped.copy()
                
            X.append(x_clipped)
            Y.append(fx)
            
            if calls >= budget:
                raise Exception("Budget exceeded")
                
            return fx

        # Initialization
        rng = np.random.default_rng(42)
        n_init = 20
        for _ in range(n_init):
            x = rng.uniform(-5.0, 5.0, dim)
            try:
                wrapped_func(x)
            except Exception:
                break

        if calls >= budget or time.time() > deadline:
            return best_x, best_fx

        # Pre-compute for GP
        X_arr = np.array(X)
        Y_arr = np.array(Y)
        K = rbf_kernel(X_arr, X_arr)
        K += noise * np.eye(len(X_arr))
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(K, Y_arr)
        except np.linalg.LinAlgError:
            # Fallback
            alpha = np.linalg.lstsq(K, Y_arr, rcond=None)[0]
            L = None

        # Loop
        while calls < budget and time.time() < deadline:
            # Generate candidates
            n_cand = 500
            X_cand = rng.uniform(-5.0, 5.0, (n_cand, dim))
            
            # Predict
            K_new = rbf_kernel(X_cand, X_arr)
            
            if L is not None:
                # Efficient prediction using Cholesky
                # mean = K_new @ alpha
                mean = K_new @ alpha
                # var = sigma^2 - sum( (K_new @ L^-1)^2 )
                # Solve L v = K_new^T  => v = L^-1 K_new^T
                # var = sigma^2 - sum(v^2, axis=0)
                v = np.linalg.solve(L, K_new.T)
                var = sigma**2 - np.sum(v**2, axis=0)
            else:
                mean = K_new @ alpha
                K_new_sq = rbf_kernel(X_cand, X_cand)
                var = np.diag(K_new_sq) - np.sum(K_new**2 * alpha[None, :], axis=1)
            
            var = np.maximum(var, 1e-10)
            
            # EI
            sigma_x = np.sqrt(var)
            z = (best_fx - mean) / sigma_x
            z = np.where(sigma_x > 1e-10, z, 0)
            ei = (best_fx - mean) * norm.cdf(z) + sigma_x * norm.pdf(z)
            
            idx = np.argmax(ei)
            x_next = X_cand[idx]
            
            try:
                wrapped_func(x_next)
            except Exception:
                break
            
            # Update GP
            # Append new point
            X_arr = np.vstack([X_arr, x_next.reshape(1, -1)])
            Y_arr = np.append(Y_arr, wrapped_func.__wrapped__ if hasattr(wrapped_func, '__wrapped__') else X[-1]) # Wait, Y is updated in wrapped_func
            # Actually, X and Y are lists. X_arr and Y_arr are arrays.
            # Let's just recompute K and alpha. It's simpler and $N^3$ is fast enough.
            X_arr = np.array(X)
            Y_arr = np.array(Y)
            K = rbf_kernel(X_arr, X_arr)
            K += noise * np.eye(len(X_arr))
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(K, Y_arr)
            except np.linalg.LinAlgError:
                alpha = np.linalg.lstsq(K, Y_arr, rcond=None)[0]
                L = None

        return best_x, best_fx
