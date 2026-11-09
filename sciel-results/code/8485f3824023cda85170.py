"""Multi-objective optimizer on RE21 using ParEGO (Bayesian Optimization)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    """ParEGO Multi-Objective Bayesian Optimizer."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        
        # 1. Initialization
        n_init = min(20, self.budget)
        X_init = lo + span * rng.random((n_init, self.dim))
        F_init = np.array([problem(x) for x in X_init], dtype=float)
        
        X_all = X_init.copy()
        F_all = F_init.copy()
        used = n_init
        
        # 2. GP Helper
        def fit_gp(X_train, y_train):
            # Standardize
            X_mean = X_train.mean(axis=0)
            X_std = X_train.std(axis=0)
            X_std[X_std == 0] = 1.0
            X_norm = (X_train - X_mean) / X_std
            
            y_mean = y_train.mean()
            y_std = y_train.std()
            if y_std == 0: y_std = 1.0
            y_norm = (y_train - y_mean) / y_std
            
            # RBF Kernel
            # l = 1.0 in normalized space
            # sigma_f = 1.0
            # sigma_n = 1e-6
            def rbf(X1, X2):
                sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
                sq_dists = np.maximum(sq_dists, 0)
                return np.exp(-0.5 * sq_dists)
            
            K = rbf(X_norm, X_norm) + 1e-6 * np.eye(len(X_norm))
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
            
            def predict(X_new):
                X_new_norm = (X_new - X_mean) / X_std
                k_star = rbf(X_new_norm, X_norm)
                mu = k_star @ alpha * y_std + y_mean
                v = 1.0 - np.sum((L.T @ np.linalg.solve(L, k_star.T)) * k_star.T)
                v = np.maximum(v, 1e-10)
                sigma = np.sqrt(v) * y_std
                return mu, sigma
            
            return predict

        def ei(mu, sigma, y_best):
            # y_best is the best (minimum) value seen
            # We want to minimize y. So improvement is y_best - y.
            # If y < y_best, improvement is positive.
            # EI = E[max(0, y_best - y)]
            # y ~ N(mu, sigma^2)
            # Let Z = (y_best - mu) / sigma
            # EI = (y_best - mu) * Phi(Z) + sigma * phi(Z)
            Z = (y_best - mu) / sigma
            return (y_best - mu) * norm.cdf(Z) + sigma * norm.pdf(Z)

        # 3. Main Loop
        while used < self.budget:
            # Scalarize
            # Sample lambda from Dirichlet(1, ..., 1)
            lambda_vec = rng.dirichlet(np.ones(self.n_obj))
            y_all = F_all @ lambda_vec
            y_best = np.min(y_all)
            
            # Fit GP
            predict = fit_gp(X_all, y_all)
            
            # Acquisition Function
            def acquisition(x):
                x = np.array(x).reshape(1, -1)
                mu, sigma = predict(x)
                return -ei(mu[0], sigma[0], y_best) # Minimize negative EI
            
            # Optimize Acquisition
            # 1. Random candidates
            n_candidates = 50
            X_cands = lo + span * rng.random((n_candidates, self.dim))
            acq_cands = np.array([acquisition(x) for x in X_cands])
            idx_best_cands = np.argsort(acq_cands)[:5]
            
            best_x = None
            best_acq = np.inf
            
            for idx in idx_best_cands:
                x0 = X_cands[idx]
                res = minimize(acquisition, x0, method='L-BFGS-B', bounds=list(zip(lo, hi)), options={'maxiter': 50})
                if res.fun < best_acq:
                    best_acq = res.fun
                    best_x = res.x
            
            # Evaluate new point
            new_x = np.clip(best_x, lo, hi)
            new_f = problem(new_x)
            
            X_all = np.vstack([X_all, new_x])
            F_all = np.vstack([F_all, new_f])
            used += 1
            
        # 4. Extract Pareto Front
        # Non-dominated sort
        n = len(F_all)
        is_dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i == j: continue
                if np.all(F_all[j] <= F_all[i]) and np.any(F_all[j] < F_all[i]):
                    is_dominated[i] = True
                    break
        
        pareto_idx = np.where(~is_dominated)[0]
        return X_all[pareto_idx]

# EVOLVE-BLOCK-END
