import numpy as np
import time
import itertools
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin
        
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        all_points = []  # (x, f) pairs
        
        def record(x, f):
            nonlocal best_x, best_f, calls_used
            calls_used += 1
            all_points.append((x.copy(), f))
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f
        
        # Phase 1: Dense grid search - 5^5 = 3125 is too many, use stratified sampling
        # Use a coarse grid first: 4^5 = 1024 points, but we only have 500 budget
        # So use 3^5 = 243 for coarse, then refine
        grid_vals = np.linspace(-5.0, 5.0, 3)
        grid_points = [np.array(combo) for combo in itertools.product(grid_vals, repeat=n)]
        
        for x in grid_points:
            if calls_used >= budget or time.time() > deadline:
                break
            fx = func(x)
            record(x, fx)
        
        if best_x is None:
            best_x = np.zeros(n)
            best_f = func(best_x)
            record(best_x, best_f)
        
        # Phase 2: Multi-start local optimization from top candidates
        # Sort by function value and take top K
        all_points.sort(key=lambda p: p[1])
        top_k = min(8, len(all_points))
        
        remaining_budget = budget - calls_used
        points_per_start = max(1, remaining_budget // top_k)
        
        for i in range(top_k):
            if calls_used >= budget or time.time() > deadline:
                break
            
            x0 = all_points[i][0].copy()
            
            # Local Nelder-Mead optimization with limited evaluations
            local_calls = [0]
            local_budget = points_per_start
            
            def local_func(x):
                if local_calls[0] >= local_budget or time.time() > deadline:
                    return best_f
                local_calls[0] += 1
                x = np.clip(x, -5, 5)
                return record(x, func(x))
            
            try:
                res = minimize(local_func, x0, method='Nelder-Mead',
                              options={'maxiter': local_budget, 'fatol': 1e-10, 'xatol': 1e-8})
            except:
                pass
        
        # Phase 3: Bayesian Optimization with simple distance-based surrogate
        # For the remaining budget, use EI-based acquisition
        remaining = budget - calls_used
        if remaining > 5 and len(all_points) >= 2:
            # Build simple surrogate model
            X = np.array([p[0] for p in all_points])
            Y = np.array([p[1] for p in all_points])
            
            # Normalize
            Y_mean = Y.mean()
            Y_std = Y.std() + 1e-10
            Y_norm = (Y - Y_mean) / Y_std
            
            # Simple kernel: Matern-5/2
            nu = 2.5
            def matern52(X1, X2, length_scale=1.0):
                # Compute pairwise distances
                diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
                dist = np.sqrt(np.sum(diff**2, axis=2))
                z = dist / length_scale
                return np.exp(-z * np.sqrt(3)) * (1 + z * np.sqrt(3))
            
            # Optimize length scale roughly
            best_ls = 1.0
            best_score = -np.inf
            for ls in [0.5, 1.0, 2.0, 3.0]:
                K = matern52(X, X, ls) + 0.1 * np.eye(len(X))
                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_norm))
                    logdet = 2 * np.sum(np.log(np.diag(L)))
                    score = -0.5 * Y_norm @ alpha - 0.5 * logdet - len(X) * np.log(2 * np.pi) / 2
                    if score > best_score:
                        best_score = score
                        best_ls = ls
                except:
                    pass
            
            K = matern52(X, X, best_ls) + 0.1 * np.eye(len(X))
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_norm))
            except:
                L = np.linalg.cholesky(K + 0.01 * np.eye(len(X)))
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_norm))
            
            # Acquisition: Expected Improvement
            f_min = Y.min()
            
            def predict(x_new):
                k_star = matern52(X, x_new.reshape(1, -1), best_ls).flatten()
                mu = k_star @ alpha
                v = np.linalg.solve(L, k_star)
                var = 1.0 - k_star @ v
                var = max(var, 1e-10)
                return mu * Y_std + Y_mean, np.sqrt(var) * Y_std
            
            # Sample candidates and pick best EI
            ei_candidates = rng.uniform(-5, 5, size=(200, n))
            ei_values = []
            for xc in ei_candidates:
                xc = np.clip(xc, -5, 5)
                mu, sigma = predict(xc)
                if sigma < 1e-10:
                    ei = 0.0
                else:
                    z = (f_min - mu) / sigma
                    from scipy.stats import norm
                    ei = (f_min - mu) * norm.cdf(z) + sigma * norm.pdf(z)
                ei_values.append(ei)
            
            ei_values = np.array(ei_values)
            # Pick top few candidates and evaluate
            top_ei_idx = np.argsort(ei_values)[-min(3, remaining):][::-1]
            
            for idx in top_ei_idx:
                if calls_used >= budget or time.time() > deadline:
                    break
                xc = ei_candidates[idx]
                fx = func(xc)
                record(xc, fx)
        
        return best_x, best_f
