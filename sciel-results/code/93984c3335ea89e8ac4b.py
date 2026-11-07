import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Bayesian Optimization with Gaussian Process and Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget

        # Data storage
        X_data = []
        Y_data = []
        best_x = None
        best_fx = np.inf
        calls = 0

        # GP Parameters (Fixed for simplicity and speed)
        length_scales = np.ones(dim)
        variance = 1.0
        noise = 1e-8

        def kernel(X1, X2):
            # X1: (n1, d), X2: (n2, d)
            # RBF Kernel
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            # Scale by length scales
            diff_scaled = diff / length_scales
            sq_dist = np.sum(diff_scaled**2, axis=2)
            return variance * np.exp(-0.5 * sq_dist)

        def fit_gp(X, Y):
            X = np.array(X)
            Y = np.array(Y)
            n = len(X)
            mean_y = np.mean(Y)
            Y_c = Y - mean_y
            K = kernel(X, X)
            K += noise * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_c))
            except np.linalg.LinAlgError:
                K += 1e-6 * np.eye(n)
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_c))
            return mean_y, L, alpha, X

        def predict_gp(X_new, mean_y, L, alpha, X_train):
            X_new = np.array(X_new)
            if X_new.ndim == 1:
                X_new = X_new.reshape(1, -1)
            K_s = kernel(X_train, X_new)
            K_ss = kernel(X_new, X_new)
            mu = K_s.T @ alpha
            var = np.diag(K_ss) - np.sum(K_s**2, axis=0)
            var = np.maximum(var, 1e-10)
            return mean_y + mu, np.sqrt(var)

        # Phase 1: Bayesian Optimization
        bo_budget = int(budget * 0.7)
        polish_budget = budget - bo_budget

        # Initial random sampling
        n_init = 10
        for _ in range(n_init):
            if calls >= bo_budget or time.time() >= deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls += 1
            X_data.append(x)
            Y_data.append(fx)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # BO Loop
        while calls < bo_budget and time.time() < deadline:
            if len(X_data) < 2:
                break
            
            # Fit GP
            try:
                mean_y, L, alpha, X_train = fit_gp(X_data, Y_data)
            except Exception:
                # If GP fails, fallback to random
                x = rng.uniform(-5, 5, dim)
                fx = func(x)
                calls += 1
                X_data.append(x)
                Y_data.append(fx)
                if fx < best_fx:
                    best_fx = fx
                    best_x = x.copy()
                continue

            # Define EI function
            f_best = best_fx
            xi = 0.01 # Exploration parameter
            
            def ei_func(x):
                x = np.clip(x, -5, 5)
                mu, sigma = predict_gp(x, mean_y, L, alpha, X_train)
                if sigma < 1e-10:
                    return 0.0
                z = (f_best - mu) / sigma
                from scipy.stats import norm
                ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
                return -ei # Minimize negative EI

            # Maximize EI using Nelder-Mead
            # Start from a few random points or the current best
            # To save time, limit NM iterations
            best_ei_x = None
            best_ei_val = np.inf
            
            # Try a few starting points for EI maximization
            starts = [best_x]
            for _ in range(3):
                starts.append(rng.uniform(-5, 5, dim))
            
            for start in starts:
                if time.time() >= deadline:
                    break
                try:
                    res = minimize(ei_func, start, method='Nelder-Mead', 
                                   options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-6})
                    if res.fun < best_ei_val:
                        best_ei_val = res.fun
                        best_ei_x = res.x
                except Exception:
                    pass

            if best_ei_x is None:
                # Fallback
                x = rng.uniform(-5, 5, dim)
            else:
                x = best_ei_x

            # Evaluate
            fx = func(x)
            calls += 1
            X_data.append(x)
            Y_data.append(fx)
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()

        # Phase 2: Nelder-Mead Polish
        remaining = budget - calls
        if remaining > 3 and time.time() < deadline:
            nm_calls = [0]
            
            def nm_func(z):
                if time.time() >= deadline:
                    raise Exception("timeout")
                if nm_calls[0] >= remaining:
                    raise Exception("budget exceeded")
                nm_calls[0] += 1
                z = np.clip(z, -5, 5)
                return func(z)

            try:
                result = minimize(
                    nm_func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': remaining,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'maxfev': remaining
                    }
                )
                if result.fun < best_fx:
                    best_x = result.x
                    best_fx = result.fun
            except Exception:
                pass

        return best_x, best_fx
