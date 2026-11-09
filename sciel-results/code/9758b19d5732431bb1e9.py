"""Bayesian Optimization with Matern-5/2 GP surrogate for Weierstrass f16."""
import numpy as np
import time
from scipy.stats import norm

class Optimizer:
    """Bayesian Optimization optimizer for BBOB f16 (Weierstrass)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0
        
        # Weierstrass function has oscillatory structure.
        # Use Matern-5/2 kernel.
        # Length scales: global scale ~ 5, local scale ~ 0.5 (to capture oscillations)
        # Actually, f16 Weierstrass has frequencies up to a^k. 
        # A single length scale might struggle. Let's try a mix or a specific one.
        # For robustness, let's use a Matern-5/2 with length_scale=1.0 and noise=1e-10.
        # We might need multiple length scales, but standard BO often uses one or a few.
        # Given the rotation, the correlation structure is aligned with the rotated axes.
        
        # Hyperparameters for the GP
        # We will not do full hyperparameter optimization to save time/budget, 
        # but we can tune them manually based on the problem.
        # Weierstrass f(x) = sum_{k=0}^inf a^k cos(2 pi b^k (x + c))
        # a=0.5, b=2, c=0.5 usually.
        # The function is periodic with period 1 in each dimension (before rotation).
        # So a length scale of ~1.0 or slightly larger is appropriate.
        
        length_scale = 1.5
        noise = 1e-10
        signal_var = 1.0 # We will normalize the function values
        
        # State
        X = []
        y = []
        best_x = None
        best_f = np.inf
        
        def matern52_kernel(X1, X2, ls, sv):
            # X1: (n, d), X2: (m, d)
            # Returns (n, m) kernel matrix
            n = X1.shape[0]
            m = X2.shape[0]
            d = X1.shape[1]
            
            # Compute distances
            # (n, 1, d) - (1, m, d) -> (n, m, d)
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            sq_dist = np.sum(diff**2, axis=2)
            
            r = np.sqrt(sq_dist) / ls
            # Matern 5/2: (1 + sqrt(5)r + 5/3 r^2) * exp(-sqrt(5)r)
            # Handle r=0 case where exp is 1 and poly is 1
            K = sv * (1 + np.sqrt(5)*r + (5/3)*r**2) * np.exp(-np.sqrt(5)*r)
            # Ensure diagonal is sv
            # np.fill_diagonal(K, sv) # Not needed if sq_dist is 0 exactly, but good for safety
            
            return K

        def update_gp(X, y, ls, sv, noise):
            """Fit GP and return mean and variance functions."""
            n = X.shape[0]
            K = matern52_kernel(X, X, ls, sv)
            K[np.diag_indices(n)] += noise
            
            # Cholesky decomposition
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            
            def predict(X_new):
                K_s = matern52_kernel(X_new, X, ls, sv) # (n_new, n)
                mean = K_s @ alpha
                # Variance
                v = np.linalg.solve(L, K_s.T) # (n, n_new)
                var = sv - np.sum(v**2, axis=0)
                # Clip variance to be non-negative
                var = np.maximum(var, 1e-10)
                return mean, var
            
            return predict

        # 1. Initialization: Latin Hypercube or Random
        # 20 initial points is a good start.
        n_init = 20
        if budget >= n_init:
            # Use a simple random initialization
            X_init = np.random.uniform(-5.0, 5.0, size=(n_init, dim))
            # Ensure we cover the space reasonably well.
            # Maybe use a grid for the first few? 
            # Let's stick to random for simplicity, or Halton.
            # Random is fine with 20 points in 5D.
            
            for i in range(n_init):
                if time.time() > deadline or budget <= 0:
                    break
                x = X_init[i]
                fx = func(x)
                budget -= 1
                X.append(x)
                y.append(fx)
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
        
        if not X:
            # Fallback
            x = np.zeros(dim)
            fx = func(x)
            return x, fx

        X = np.array(X)
        y = np.array(y)
        
        # Normalize y for GP stability? 
        # GP is invariant to scaling of y if we also scale kernel, but 
        # numerically it's better to standardize.
        y_mean = np.mean(y)
        y_std = np.std(y)
        if y_std < 1e-10:
            y_std = 1.0
        y_norm = (y - y_mean) / y_std
        
        # We will re-fit the GP every few steps to incorporate new data.
        # To save computation, we can just re-fit every time. 
        # With n=500, d=5, K is 500x500. Cholesky is O(n^3) ~ 125M ops. 
        # This might be slow in Python if not optimized, but NumPy is C-backed.
        # 500^3 = 1.25e8. Should be fast enough (milliseconds).
        
        # Acquisition Function: Expected Improvement (EI)
        def expected_improvement(X_new, predict_func, best_y):
            # X_new: (n_cand, d)
            # predict_func returns mean (n_cand,) and var (n_cand,)
            mean, var = predict_func(X_new)
            std = np.sqrt(var)
            
            # Avoid division by zero
            std_safe = np.where(std > 1e-10, std, 1e-10)
            
            # EI formula
            # If mean < best_y, improvement is 0
            # Else, EI = (best_y - mean) * Phi(Z) + std * phi(Z)
            # where Z = (best_y - mean) / std
            # Note: We are minimizing, so "improvement" is best_y - f(x)
            
            Z = (best_y - mean) / std_safe
            # For points where mean > best_y, Z is negative, Phi(Z) is small, EI is small.
            # We should clip Z or handle it.
            
            EI = np.zeros_like(mean)
            mask = mean < best_y
            if np.any(mask):
                Z_masked = Z[mask]
                EI[mask] = (best_y - mean[mask]) * norm.cdf(Z_masked) + std[mask] * norm.pdf(Z_masked)
            
            return EI

        # Main Loop
        while budget > 0:
            if time.time() > deadline:
                break
            
            # 1. Update GP
            # Re-normalize y if necessary? 
            # For simplicity, we keep the original normalization from init.
            # As new points are added, the mean/std might change.
            # Refitting with new normalization is better.
            y_mean = np.mean(y)
            y_std = np.std(y)
            if y_std < 1e-10:
                y_std = 1.0
            y_norm = (y - y_mean) / y_std
            
            # We need to map the kernel output to the original scale?
            # No, the GP is fit on y_norm. 
            # The predicted mean/var are in the normalized space.
            # To compare with best_f (original scale), we must transform back.
            # mean_orig = mean_norm * y_std + y_mean
            # var_orig = var_norm * y_std^2
            
            predict_func_norm = update_gp(X, y_norm, length_scale, signal_var, noise)
            
            def predict_func_orig(X_new):
                mean_n, var_n = predict_func_norm(X_new)
                mean_o = mean_n * y_std + y_mean
                var_o = var_n * y_std**2
                return mean_o, var_o
            
            # 2. Generate candidates
            # Sample 500 random points
            n_cand = 500
            X_cand = np.random.uniform(-5.0, 5.0, size=(n_cand, dim))
            
            # 3. Evaluate EI
            # best_f is the current minimum in original scale
            EI = expected_improvement(X_cand, predict_func_orig, best_f)
            
            # 4. Select best candidate
            idx = np.argmax(EI)
            x_next = X_cand[idx]
            
            # 5. Evaluate
            fx_next = func(x_next)
            budget -= 1
            
            # 6. Update state
            X = np.vstack([X, x_next])
            y = np.append(y, fx_next)
            
            if fx_next < best_f:
                best_f = fx_next
                best_x = x_next.copy()
                
            # Optional: Adapt length scale?
            # If we get stuck, maybe the length scale is wrong.
            # But changing it requires refitting.
            # For now, keep it fixed.
            
            # Check time again to be safe
            if time.time() > deadline:
                break

        return best_x, best_f
