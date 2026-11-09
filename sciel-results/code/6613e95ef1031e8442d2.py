"""Gaussian Process surrogate-guided optimizer with adaptive exploration for BBOB f16."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0

        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        n_calls = 0

        def check_time():
            return time.time() > deadline

        def call_func(x):
            nonlocal n_calls, best_x, best_f
            if n_calls >= budget:
                return None, None
            n_calls += 1
            try:
                x_clipped = np.clip(np.asarray(x, dtype=float), -5, 5)
                f_val = float(func(x_clipped))
                if f_val < best_f:
                    best_f = f_val
                    best_x = x_clipped.copy()
                return f_val, x_clipped
            except Exception:
                return None, None

        rng = np.random.default_rng(42)

        # Phase 1: Space-filling initial sampling (Sobol-like via random + jitter)
        n_init = 20
        X_obs = []
        Y_obs = []
        for _ in range(n_init):
            if check_time() or n_calls >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            f_val, x_eval = call_func(x)
            if f_val is not None:
                X_obs.append(x_eval)
                Y_obs.append(f_val)

        if best_x is None:
            x = np.zeros(dim)
            call_func(x)
            if best_x is None:
                return None

        X_obs = np.array(X_obs)
        Y_obs = np.array(Y_obs)

        # GP kernel: Matern 5/2 with adaptive length scale
        def matern_kernel(X1, X2, length_scale, variance):
            """Compute Matern 5/2 kernel matrix."""
            d = np.sqrt(np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=-1))
            sqrt5_d = np.sqrt(5.0) * d / length_scale
            K = variance * (1.0 + sqrt5_d + sqrt5_d ** 2 / 3.0) * np.exp(-sqrt5_d)
            return K

        def predict_gp(X_new, X_train, Y_train, length_scale, variance, noise):
            """GP prediction: mean and variance at X_new."""
            K = matern_kernel(X_train, X_train, length_scale, variance)
            K[np.diag_indices_from(K)] += noise
            K_new = matern_kernel(X_train, X_new, length_scale, variance)
            
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
                mean = K_new.T @ alpha
                v = np.linalg.solve(L, K_new)
                var = variance - np.sum(v ** 2, axis=0)
                var = np.maximum(var, 1e-10)
            except np.linalg.LinAlgError:
                # Fallback to ridge regression
                reg = 1e-6
                K_reg = K + reg * np.eye(len(X_train))
                try:
                    alpha = np.linalg.solve(K_reg, Y_train)
                    mean = K_new.T @ alpha
                    var = np.full(len(X_new), variance)
                except np.linalg.LinAlgError:
                    mean = np.full(len(X_new), np.mean(Y_train))
                    var = np.full(len(X_new), np.var(Y_train))
            
            return mean, var

        # Phase 2: GP-guided optimization with adaptive exploration
        length_scale = 1.0
        variance = 1.0
        noise = 1e-6
        
        # Normalize Y for stable GP
        Y_mean = np.mean(Y_obs)
        Y_std = np.std(Y_obs) + 1e-10
        Y_norm = (Y_obs - Y_mean) / Y_std

        # Adaptive parameters
        exploit_weight = 0.3  # Start with more exploration
        max_gp_updates = 30
        gp_update_interval = 5  # Update GP every N new points
        
        stagnation_count = 0
        best_f_prev = best_f

        # Phase 2a: GP-guided acquisition
        while n_calls < budget and not check_time():
            # Update GP parameters adaptively
            if len(X_obs) >= 10:
                # Estimate length scale from data spread
                X_range = np.ptp(X_obs, axis=0)
                length_scale = np.max(X_range) / (2.0 * np.sqrt(dim))
                length_scale = np.clip(length_scale, 0.1, 5.0)
                
                # Estimate variance
                variance = np.var(Y_norm) + 1e-6
                noise = 1e-5 * (np.max(np.abs(Y_norm)) + 1.0)

            # Generate candidate points using multiple strategies
            candidates = []
            
            # Strategy 1: Expected Improvement on a grid around best
            n_grid = 50
            for _ in range(n_grid):
                # Sample around best point with decreasing radius
                radius = 0.5 + 2.0 * np.exp(-0.1 * len(X_obs))
                x_cand = best_x + rng.normal(0, radius, dim)
                x_cand = np.clip(x_cand, -5, 5)
                candidates.append(x_cand)
            
            # Strategy 2: Pure random for exploration
            for _ in range(10):
                x_cand = rng.uniform(-5, 5, dim)
                candidates.append(x_cand)
            
            # Strategy 3: Latin hypercube-like spread
            for _ in range(10):
                x_cand = rng.uniform(-5, 5, dim)
                # Perturb slightly
                x_cand += rng.normal(0, 0.5, dim)
                x_cand = np.clip(x_cand, -5, 5)
                candidates.append(x_cand)
            
            candidates = np.array(candidates)
            
            # Compute GP predictions and acquisition function
            if len(X_obs) >= 5:
                X_train = X_obs
                Y_train = Y_norm
            else:
                X_train = X_obs
                Y_train = Y_obs
            
            mean, var = predict_gp(candidates, X_train, Y_train, length_scale, variance, noise)
            
            # Expected Improvement acquisition
            best_y_norm = np.min(Y_norm) if len(Y_norm) > 0 else 0.0
            Z = (best_y_norm - mean) / np.sqrt(var)
            from scipy.stats import norm
            EI = (best_y_norm - mean) * norm.cdf(Z) + np.sqrt(var) * norm.pdf(Z)
            EI = np.maximum(EI, 0)
            
            # Mix exploitation and exploration
            # Use quantile-based selection
            n_select = min(3, len(EI))
            # Top EI candidates
            top_ei_idx = np.argsort(EI)[-n_select:]
            
            # Also add some random candidates for diversity
            n_random = max(1, len(EI) // 5)
            random_idx = rng.choice(len(EI), n_random, replace=False)
            
            selected_idx = np.unique(np.concatenate([top_ei_idx, random_idx]))
            
            # Evaluate selected candidates
            for idx in selected_idx:
                if check_time() or n_calls >= budget:
                    break
                x_trial = candidates[idx]
                f_val, x_eval = call_func(x_trial)
                if f_val is not None:
                    X_obs = np.vstack([X_obs, x_eval])
                    Y_obs = np.append(Y_obs, f_val)
                    Y_norm = (Y_obs - Y_mean) / Y_std
                    
                    if f_val < best_f_prev:
                        best_f_prev = f_val
                        stagnation_count = 0
                    else:
                        stagnation_count += 1
            
            # Adaptive: if stagnating, increase exploration
            if stagnation_count > 8:
                # Reset exploration: sample more randomly
                for _ in range(5):
                    if check_time() or n_calls >= budget:
                        break
                    x_trial = rng.uniform(-5, 5, dim)
                    f_val, x_eval = call_func(x_trial)
                    if f_val is not None:
                        X_obs = np.vstack([X_obs, x_eval])
                        Y_obs = np.append(Y_obs, f_val)
                        Y_norm = (Y_obs - Y_mean) / Y_std
                stagnation_count = 0

        # Phase 3: Local refinement with L-BFGS-B around best point
        if n_calls < budget and not check_time():
            # Use scipy minimize with numerical gradient
            best_x_local = best_x.copy()
            
            def objective(x):
                nonlocal n_calls
                if n_calls >= budget:
                    return np.inf
                x_clipped = np.clip(x, -5, 5)
                f_val, _ = call_func(x_clipped)
                return f_val if f_val is not None else np.inf
            
            # Multiple local starts
            for _ in range(3):
                if check_time() or n_calls >= budget:
                    break
                
                x0 = best_x + rng.normal(0, 0.1, dim)
                x0 = np.clip(x0, -5, 5)
                
                try:
                    res = minimize(objective, x0, method='L-BFGS-B', 
                                  bounds=[(-5, 5)] * dim,
                                  options={'maxiter': 20, 'maxfun': 30})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
