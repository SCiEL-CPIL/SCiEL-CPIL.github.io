"""Hybrid GP-Bayesian Optimization with Local Search for BBOB f19."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import time


class Optimizer:
    """GP-based Bayesian Optimizer with Powell local refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        
        # 1. Initialization: Latin Hypercube Sampling
        n_init = 15
        X_init = np.zeros((n_init, self.dim))
        for i in range(self.dim):
            perm = rng.permutation(n_init)
            X_init[perm, i] = np.linspace(-5, 5, n_init)
        
        Y_init = np.array([func(x) for x in X_init])
        
        X = X_init
        Y = Y_init
        best_x = X_init[np.argmin(Y_init)]
        best_y = Y_init.min()
        
        calls = n_init
        
        # GP Configuration
        # RBF Kernel with fixed length scale
        L_SCALE = 2.0
        
        def compute_kernel(X1, X2):
            """Compute RBF kernel matrix."""
            sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * (X1 @ X2.T)
            sq_dists = np.maximum(sq_dists, 0)
            return np.exp(-0.5 * sq_dists / (L_SCALE**2))
        
        def fit_gp(X, Y):
            """Fit GP to normalized Y. Returns L, alpha, y_min, y_range."""
            y_min = Y.min()
            y_max = Y.max()
            y_range = y_max - y_min
            if y_range < 1e-8:
                y_range = 1.0
            Y_norm = (Y - y_min) / y_range
            
            K = compute_kernel(X, X)
            K += 1e-8 * np.eye(len(X))  # Jitter for stability
            try:
                L = np.linalg.cholesky(K)
                v = np.linalg.solve(L, Y_norm)
                alpha = np.linalg.solve(L.T, v)
                return L, alpha, y_min, y_range
            except np.linalg.LinAlgError:
                return None, None, y_min, y_range

        def ei_function(x, X, Y, L, alpha, y_min, y_range, best_y_norm):
            """Calculate Expected Improvement for a single point x."""
            x = x.reshape(1, -1)
            K_x = compute_kernel(X, x)  # (n, 1)
            K_xx = compute_kernel(x, x)  # (1, 1)
            
            # Mean in normalized space
            mu_norm = (K_x.T @ alpha).item()
            
            # Variance in normalized space
            v = np.linalg.solve(L, K_x.flatten())
            var = K_xx.item() - (v.T @ v)
            var = max(var, 1e-10)
            std = np.sqrt(var)
            
            if std < 1e-8:
                return 0.0
            
            z = (mu_norm - best_y_norm) / std
            # EI in normalized space
            ei = (mu_norm - best_y_norm) * norm.cdf(z) + std * norm.pdf(z)
            return ei

        # Main Loop
        while calls < self.budget and (time.time() - start_time) < deadline:
            # Fit GP
            fit_result = fit_gp(X, Y)
            if fit_result[0] is None:
                # Fallback: Random point
                x_new = rng.uniform(-5, 5, self.dim)
                y_new = func(x_new)
                calls += 1
                X = np.vstack([X, x_new])
                Y = np.append(Y, y_new)
                if y_new < best_y:
                    best_y = y_new
                    best_x = x_new
                continue
            
            L, alpha, y_min, y_range = fit_result
            best_y_norm = (best_y - y_min) / y_range
            
            # Maximize EI using multi-start L-BFGS-B
            best_ei = -np.inf
            best_x_new = None
            
            # Generate starts
            starts = [best_x + 0.5 * rng.standard_normal(self.dim)]
            for _ in range(3):
                starts.append(rng.uniform(-5, 5, self.dim))
            
            for start in starts:
                try:
                    res = minimize(
                        lambda x: -ei_function(x, X, Y, L, alpha, y_min, y_range, best_y_norm),
                        start,
                        method='L-BFGS-B',
                        bounds=[(-5, 5)] * self.dim,
                        options={'maxiter': 50}
                    )
                    if res.success:
                        ei_val = -res.fun
                        if ei_val > best_ei:
                            best_ei = ei_val
                            best_x_new = res.x
                except Exception:
                    pass
            
            if best_x_new is None:
                best_x_new = rng.uniform(-5, 5, self.dim)
            
            # Evaluate new point
            y_new = func(best_x_new)
            calls += 1
            
            # Update data
            X = np.vstack([X, best_x_new])
            Y = np.append(Y, y_new)
            
            if y_new < best_y:
                best_y = y_new
                best_x = best_x_new

        # Final Polish: Powell Local Search
        remaining = self.budget - calls
        if remaining > 5 and (time.time() - start_time) < deadline - 10:
            count = [0]
            def counted_func(x):
                count[0] += 1
                if count[0] > remaining:
                    raise Exception("Budget exceeded")
                return func(x)
            
            try:
                max_iter = max(1, remaining // self.dim)
                res = minimize(
                    counted_func,
                    best_x,
                    method='Powell',
                    options={'maxiter': max_iter, 'xtol': 1e-6, 'ftol': 1e-6}
                )
                if res.fun < best_y:
                    best_y = res.fun
                    best_x = res.x
            except Exception:
                pass

        return best_x, best_y
# EVOLVE-BLOCK-END
