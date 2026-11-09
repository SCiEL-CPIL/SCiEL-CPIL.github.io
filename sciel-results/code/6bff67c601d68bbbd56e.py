    import numpy as np
    import time
    from scipy.stats import norm
    from scipy.optimize import minimize

    class Optimizer:
        def __init__(self, budget, dim):
            self.budget = budget
            self.dim = dim
            self.rng = np.random.default_rng(42)
            self.X = []
            self.y = []
            self.evals = 0
            self.best_x = None
            self.best_f = float('inf')
            self.deadline = 0

        def _fit_gp(self, X, y, length_scale=1.0, sigma_f=1.0, jitter=1e-6):
            # Standardize y
            y_mean = np.mean(y)
            y_std = np.std(y)
            if y_std < 1e-8: y_std = 1.0
            y_norm = (y - y_mean) / y_std
            
            n = len(X)
            d = X.shape[1]
            
            # Matern 5/2 kernel
            def kernel(x1, x2):
                # Compute pairwise distances
                # x1: (n, d), x2: (m, d)
                x1_norm = np.sum(x1**2, axis=1)[:, None]
                x2_norm = np.sum(x2**2, axis=1)[None, :]
                dist_sq = x1_norm + x2_norm - 2 * x1 @ x2.T
                dist_sq = np.maximum(dist_sq, 0)
                dist = np.sqrt(dist_sq)
                # Matern 5/2
                sqrt_5 = np.sqrt(5.0)
                r = dist / length_scale
                K = (1 + r * sqrt_5 + (5/3) * r**2) * np.exp(-r * sqrt_5)
                return K
            
            K = kernel(X, X)
            K += jitter * np.eye(n)
            
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
                # Store L, alpha, y_mean, y_std
                return L, alpha, y_mean, y_std, length_scale
            except np.linalg.LinAlgError:
                # Fallback: return identity-like structure?
                # If Cholesky fails, add more jitter
                K += 1e-4 * np.eye(n)
                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
                    return L, alpha, y_mean, y_std, length_scale
                except:
                    return None

        def _predict_gp(self, x_new, gp_params):
            if gp_params is None:
                return 0, 1
            L, alpha, y_mean, y_std, length_scale = gp_params
            X_train = self.X
            d = self.dim
            
            # Kernel between x_new and X_train
            def kernel_vec(x1, x2):
                dist_sq = np.sum((x1 - x2)**2)
                dist = np.sqrt(max(dist_sq, 0))
                sqrt_5 = np.sqrt(5.0)
                r = dist / length_scale
                return (1 + r * sqrt_5 + (5/3) * r**2) * np.exp(-r * sqrt_5)
            
            k_star = np.array([kernel_vec(x_new, x) for x in X_train])
            
            # Mean
            mu = np.dot(k_star, alpha) * y_std + y_mean
            
            # Variance
            v = np.linalg.solve(L, k_star)
            var = 1.0 - np.dot(k_star, v)
            var = max(var, 1e-8)
            sigma = np.sqrt(var) * y_std
            
            return mu, sigma

        def _expected_improvement(self, x, gp_params, y_best, xi=0.01):
            mu, sigma = self._predict_gp(x, gp_params)
            if sigma < 1e-8:
                return max(0, y_best - mu)
            z = (y_best - mu) / sigma
            ei = (y_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return max(0, ei - xi * sigma) # Simplified EI

        def __call__(self, func):
            t0 = time.time()
            self.deadline = t0 + 290 # 40s buffer
            
            dim = self.dim
            budget = self.budget
            
            # 1. Initial Sampling
            n_init = 15
            self.X = []
            self.y = []
            
            for _ in range(n_init):
                if self.evals >= budget or time.time() > self.deadline:
                    break
                x = self.rng.uniform(-4, 4, dim)
                x = np.clip(x, -5, 5)
                y = func(x)
                if np.isnan(y) or np.isinf(y):
                    y = 1e6
                self.X.append(x)
                self.y.append(y)
                self.evals += 1
                if y < self.best_f:
                    self.best_f = y
                    self.best_x = x.copy()
            
            if not self.X:
                return np.zeros(dim), 0.0
            
            self.X = np.array(self.X)
            self.y = np.array(self.y)
            
            # 2. BO Loop
            while self.evals < budget and time.time() < self.deadline:
                # Fit GP
                gp_params = self._fit_gp(self.X, self.y)
                if gp_params is None:
                    # If GP fails, fall back to random search
                    x = self.rng.uniform(-4, 4, dim)
                else:
                    # Optimize EI
                    # Multi-start Nelder-Mead for EI
                    best_ei = -np.inf
                    best_x_ei = None
                    y_best = self.best_f
                    
                    # 10 random starts for EI optimization
                    for _ in range(10):
                        x0 = self.rng.uniform(-4, 4, dim)
                        # Define objective to minimize: -EI
                        def objective(x):
                            return -self._expected_improvement(x, gp_params, y_best)
                        
                        try:
                            res = minimize(objective, x0, method='Nelder-Mead', 
                                           options={'maxiter': 50, 'xatol': 0.1, 'fatol': 1e-4})
                            if res.success or res.fun < best_ei:
                                best_ei = res.fun
                                best_x_ei = res.x
                        except:
                            pass
                    
                    if best_x_ei is None:
                        x = self.rng.uniform(-4, 4, dim)
                    else:
                        x = best_x_ei
                
                x = np.clip(x, -5, 5)
                y = func(x)
                if np.isnan(y) or np.isinf(y):
                    y = 1e6
                
                self.X = np.vstack([self.X, x])
                self.y = np.append(self.y, y)
                self.evals += 1
                
                if y < self.best_f:
                    self.best_f = y
                    self.best_x = x.copy()
                
                # Adaptive Length Scale?
                # For simplicity, keep length_scale fixed at 1.0 or adapt slowly.
                # If we want to adapt, we can optimize hyperparams, but that's expensive.
                # Let's stick to fixed LS=1.0 for stability, or maybe 0.5.
            
            return self.best_x, self.best_f
    