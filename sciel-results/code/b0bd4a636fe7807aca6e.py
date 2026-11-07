import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.global_best = np.inf
        self.func_calls = 0
        self.X = []
        self.y = []
        self.start_time = None

    def __call__(self, func):
        self.start_time = time.time()
        deadline = self.start_time + 280.0 # Safety margin
        
        # 1. Initialization
        n_init = 10
        rng = np.random.default_rng(42)
        X_init = rng.uniform(-5, 5, (n_init, self.dim))
        
        for x in X_init:
            if self.func_calls >= self.budget:
                break
            y = func(x)
            if y < self.global_best:
                self.global_best = y
            self.X.append(x)
            self.y.append(y)
            self.func_calls += 1

        # 2. BO Loop
        l = 1.0
        s2 = 1.0
        ns2 = 1e-6
        
        while self.func_calls < self.budget:
            if time.time() > deadline:
                break
            
            X = np.array(self.X)
            y = np.array(self.y)
            n = len(y)
            
            # GP Fit
            X_norm = np.sum(X**2, axis=1)
            D2 = X_norm[:, None] + X_norm[None, :] - 2 * X @ X.T
            D2 = np.maximum(D2, 0)
            K = s2 * np.exp(-0.5 * D2 / (l**2))
            K[np.diag_indices_from(K)] += ns2
            
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
            except np.linalg.LinAlgError:
                K_reg = K + 1e-6 * np.eye(n)
                alpha = np.linalg.solve(K_reg, y)

            # Acquisition Function
            def acquisition(x):
                x = np.array(x)
                x_norm = np.sum(x**2)
                D2_pred = x_norm + X_norm - 2 * x @ X.T
                D2_pred = np.maximum(D2_pred, 0)
                K_pred = s2 * np.exp(-0.5 * D2_pred / (l**2))
                
                mu = K_pred @ alpha
                v = np.linalg.solve(L, K_pred)
                var = s2 - np.sum(K_pred * v)
                var = max(var, 1e-12)
                sigma = np.sqrt(var)
                
                y_best = self.global_best
                if sigma < 1e-8:
                    return 0.0
                Z_val = (y_best - mu) / sigma
                EI = (y_best - mu) * norm.cdf(Z_val) + sigma * norm.pdf(Z_val)
                return -EI

            # Maximize Acquisition
            # Use 3 random starting points for Nelder-Mead
            best_x_next = None
            best_acq_val = np.inf
            
            for _ in range(3):
                x0 = rng.uniform(-5, 5, self.dim)
                res = minimize(acquisition, x0, method='Nelder-Mead', 
                               options={'xatol': 1e-3, 'fatol': 1e-6, 'maxiter': 50})
                if res.success or res.fun < best_acq_val:
                    best_acq_val = res.fun
                    best_x_next = res.x
            
            if best_x_next is None:
                best_x_next = rng.uniform(-5, 5, self.dim)
                
            x_next = np.clip(best_x_next, -5, 5)
            
            y_next = func(x_next)
            if y_next < self.global_best:
                self.global_best = y_next
            self.X.append(x_next)
            self.y.append(y_next)
            self.func_calls += 1

        return self.global_best
