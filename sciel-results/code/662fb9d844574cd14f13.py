import numpy as np
import time
from scipy.stats import norm
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = np.zeros(dim)
        best_f = float('inf')
        calls_made = 0

        # Data storage
        X_obs = []
        y_obs = []

        def time_ok():
            return time.time() < deadline

        def call_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget or not time_ok():
                return None
            x = np.clip(x, -5.0, 5.0)
            f_val = func(x)
            calls_made += 1
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            X_obs.append(x)
            y_obs.append(f_val)
            return f_val

        # Phase 1: Initial Sampling
        n_init = 20
        # Latin Hypercube Sampling
        lxs = []
        for i in range(n_init):
            x = np.zeros(dim)
            for d in range(dim):
                u = rng.random()
                x[d] = (u + rng.random()) / n_init * 10.0 - 5.0
            lxs.append(x)
        
        for x in lxs:
            f = call_func(x)
            if f is None:
                break

        if calls_made >= budget or not time_ok():
            return best_x, best_f

        # GP Hyperparameters
        # Matern 5/2 kernel
        # k(x, x') = sigma_f^2 * (1 + sqrt(5)*r + 5/3*r^2) * exp(-sqrt(5)*r)
        # r = ||x - x'|| / l
        sigma_f = 1.0
        length_scale = 1.0
        noise = 1e-4

        def matern52_kernel(X1, X2):
            # X1: (n1, d), X2: (n2, d)
            # Calculate pairwise distances
            # ||x - x'||^2 = ||x||^2 + ||x'||^2 - 2 x.x'
            sq_norms = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :]
            dists_sq = np.maximum(sq_norms - 2 * np.dot(X1, X2.T), 0)
            dists = np.sqrt(dists_sq) / length_scale
            k = sigma_f**2 * (1 + np.sqrt(5) * dists + 5/3 * dists**2) * np.exp(-np.sqrt(5) * dists)
            return k

        def fit_gp(X, y):
            # X: (n, d), y: (n,)
            n = len(X)
            K = matern52_kernel(X, X)
            K += noise * np.eye(n)
            # Cholesky
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # Regularize if not PD
                K += 1e-4 * np.eye(n)
                L = np.linalg.cholesky(K)
            # Solve L L^T alpha = y
            # L v = y => v = L^{-1} y
            v = np.linalg.solve(L, y)
            alpha = np.linalg.solve(L.T, v)
            return L, alpha

        def predict_gp(x_new, X, L, alpha, y_min, y_range):
            # x_new: (d,)
            # X: (n, d)
            # Predict normalized y
            k_s = matern52_kernel(x_new.reshape(1, -1), X)[:, 0] # (n,)
            mu_norm = np.dot(k_s, alpha)
            # Variance
            v = np.linalg.solve(L, k_s)
            var_norm = sigma_f**2 - np.dot(k_s, v)
            var_norm = max(var_norm, 1e-8)
            std_norm = np.sqrt(var_norm)
            
            # Denormalize
            mu = mu_norm * y_range + y_min
            std = std_norm * y_range
            return mu, std

        # Main BO Loop
        while calls_made < budget and time_ok():
            # Prepare data
            X_arr = np.array(X_obs)
            y_arr = np.array(y_obs)
            
            # Normalize y
            y_min = np.min(y_arr)
            y_max = np.max(y_arr)
            y_range = y_max - y_min + 1e-8
            y_norm = (y_arr - y_min) / y_range
            
            # Fit GP
            L, alpha = fit_gp(X_arr, y_norm)
            
            # Define EI
            f_best_norm = (best_f - y_min) / y_range
            
            def ei_func(x):
                mu, std = predict_gp(x, X_arr, L, alpha, y_min, y_range)
                mu_norm = (mu - y_min) / y_range
                std_norm = std / y_range
                
                if std_norm < 1e-8:
                    return 0.0
                
                z = (f_best_norm - mu_norm) / std_norm
                # EI = (f_best - mu) Phi(z) + std phi(z)
                # In normalized space:
                ei_norm = (f_best_norm - mu_norm) * norm.cdf(z) + std_norm * norm.pdf(z)
                return -ei_norm # Minimize negative EI

            # Optimize EI
            # Multi-start
            best_ei_val = float('inf')
            best_x_candidate = None
            
            # Starts: best_x, random points
            starts = [best_x]
            for _ in range(5):
                starts.append(rng.uniform(-5, 5, dim))
            
            for x0 in starts:
                res = minimize(ei_func, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-6})
                if res.fun < best_ei_val:
                    best_ei_val = res.fun
                    best_x_candidate = res.x
            
            if best_x_candidate is None:
                best_x_candidate = rng.uniform(-5, 5, dim)
                
            best_x_candidate = np.clip(best_x_candidate, -5.0, 5.0)
            
            # Evaluate
            f = call_func(best_x_candidate)
            if f is None:
                break

        return best_x, best_f
