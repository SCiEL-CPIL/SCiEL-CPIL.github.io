"""Katsuura (f23) minimizer using Bayesian Optimization with Gaussian Process.

Strategy:
  1. Initial coarse sampling to identify promising regions.
  2. Bayesian Optimization loop:
     - Fit a Gaussian Process surrogate model to observed data.
     - Maximize Expected Improvement (EI) acquisition function using scipy.optimize.
     - Evaluate the true function at the best surrogate point.
  3. Final polish with Nelder-Mead on the best point found.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """
    Katsuura (f23) in 5 dimensions using Bayesian Optimization.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 325.0  # safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        state = {'calls': 0, 'best_x': None, 'best_f': float('inf')}
        rng = np.random.default_rng(42)
        X_obs = []
        Y_obs = []

        def time_ok():
            return time.time() - start_time < time_limit

        def call(x):
            """Evaluate one point, update best, respect budget and time."""
            if state['calls'] >= budget or not time_ok():
                return state['best_f']
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            try:
                v = float(func(x))
            except Exception:
                return state['best_f']
            if not np.isfinite(v):
                v = state['best_f']
                return v
            
            # Check if point is already in X_obs to avoid duplicate GP fitting issues
            # (Though BO shouldn't propose exact same point, floating point might be close)
            # For strict budget tracking, we count every func call.
            state['calls'] += 1
            if v < state['best_f']:
                state['best_f'] = v
                state['best_x'] = x.copy()
            
            X_obs.append(x)
            Y_obs.append(v)
            return v

        # ---------- 1. Initial coarse sampling ----------
        # Use a mix of random and structured points
        n_init = 50
        probes = []
        probes.append(np.zeros(dim))
        for _ in range(n_init - 1):
            probes.append(rng.uniform(-5, 5, dim))
        
        for p in probes:
            if state['calls'] >= budget - 50 or not time_ok():
                break
            call(p)

        if state['best_x'] is None:
            state['best_x'] = np.zeros(dim)
            state['best_f'] = call(state['best_x'])

        best = state['best_x'].copy()
        
        # Helper to compute GP predictions
        def gp_predict(X_new):
            """
            Predict mean and variance for X_new given X_obs, Y_obs.
            Uses RBF kernel.
            """
            if len(X_obs) == 0:
                return np.zeros(X_new.shape[0]), np.ones(X_new.shape[0])
            
            X = np.array(X_obs)
            y = np.array(Y_obs)
            
            # Standardize y for numerical stability in GP
            y_mean = np.mean(y)
            y_std = np.std(y) + 1e-8
            y_stdized = (y - y_mean) / y_std
            
            # RBF Kernel
            def rbf_kernel(X1, X2, length_scale=1.0, variance=1.0):
                # Compute pairwise squared distances
                # X1: (n, d), X2: (m, d)
                X1_sq = np.sum(X1**2, axis=1, keepdims=True)
                X2_sq = np.sum(X2**2, axis=1, keepdims=True)
                dist_sq = X1_sq + X2_sq.T - 2 * X1 @ X2.T
                dist_sq = np.maximum(dist_sq, 0)
                return variance * np.exp(-0.5 * dist_sq / (length_scale**2))
            
            # Hyperparameters (fixed for simplicity, could be optimized but costs time)
            length_scale = 1.0
            variance = 1.0
            noise = 1e-4
            
            K = rbf_kernel(X, X, length_scale, variance) + noise * np.eye(len(X))
            Kf = rbf_kernel(X, X_new, length_scale, variance)
            
            try:
                # Solve K alpha = y
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_stdized))
                
                mu_stdized = Kf.T @ alpha
                v = np.linalg.solve(L, Kf.T)
                var_stdized = variance - np.sum(v**2, axis=0)
                var_stdized = np.maximum(var_stdized, 1e-8)
                
                mu = mu_stdized * y_std + y_mean
                var = var_stdized * (y_std**2)
                
            except np.linalg.LinAlgError:
                # Fallback if Cholesky fails
                try:
                    alpha = np.linalg.solve(K, y_stdized)
                    mu_stdized = Kf.T @ alpha
                    var_stdized = variance - np.sum(Kf * Kf.T @ np.linalg.solve(K, np.ones(len(X))), axis=1)
                    var_stdized = np.maximum(var_stdized, 1e-8)
                    mu = mu_stdized * y_std + y_mean
                    var = var_stdized * (y_std**2)
                except Exception:
                    mu = np.full(X_new.shape[0], y_mean)
                    var = np.full(X_new.shape[0], y_std**2)
                    
            return mu, var

        def expected_improvement(X_new, best_f):
            """
            Compute Expected Improvement for a batch of points X_new.
            """
            mu, var = gp_predict(X_new)
            sigma = np.sqrt(var)
            z = (mu - best_f) / sigma
            EI = (mu - best_f) * norm.cdf(z) + sigma * norm.pdf(z)
            # Handle case where sigma is very small
            EI[sigma < 1e-8] = np.maximum(mu[sigma < 1e-8] - best_f, 0)
            return EI

        # ---------- 2. Bayesian Optimization Loop ----------
        # We reserve some budget for final polish
        max_bo_calls = budget - 20
        
        while state['calls'] < max_bo_calls and time_ok():
            # Select points to evaluate
            # We will propose 5 candidates per iteration to have a backup if one fails
            n_candidates = 5
            candidates = []
            
            # Multi-start for EI maximization to avoid local maxima in acquisition function
            # Start from best known point and random points
            start_points = [best.copy()]
            for _ in range(4):
                start_points.append(rng.uniform(-5, 5, dim))
            
            for sp in start_points:
                if state['calls'] >= max_bo_calls or not time_ok():
                    break
                
                # Maximize EI by minimizing -EI
                # We use a grid or optimizer. Since dim=5, scipy minimize is fast.
                def neg_ei(x):
                    x = np.clip(x, -5, 5)
                    ei_val = expected_improvement(x.reshape(1, -1), state['best_f'])[0]
                    return -ei_val
                
                try:
                    res = minimize(
                        neg_ei, 
                        sp, 
                        method='Nelder-Mead', 
                        options={'maxiter': 50, 'xatol': 1e-3, 'fatol': 1e-5}
                    )
                    if res.success and np.isfinite(res.fun):
                        candidates.append(res.x)
                except Exception:
                    pass
            
            if not candidates:
                # Fallback to random if EI optimization failed
                candidates = [rng.uniform(-5, 5, dim)]
            
            # Evaluate the best candidate from the EI optimization
            # Sort candidates by EI value
            ei_scores = expected_improvement(np.array(candidates), state['best_f'])
            best_idx = np.argmax(ei_scores)
            x_next = candidates[best_idx]
            
            # Evaluate true function
            call(x_next)
            
            # Update best
            if state['best_x'] is not None:
                best = state['best_x'].copy()

        # ---------- 3. Final Polish ----------
        if state['calls'] < budget - 5 and time_ok():
            # Use Nelder-Mead to refine the best point
            try:
                res = minimize(
                    lambda x: call(x),
                    best.copy(),
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-10,
                             'adaptive': True},
                )
                if np.isfinite(res.fun) and res.fun < state['best_f']:
                    state['best_f'] = float(res.fun)
                    state['best_x'] = np.clip(np.asarray(res.x, dtype=float), -5, 5)
            except Exception:
                pass

        if state['best_x'] is None:
            state['best_x'] = np.zeros(dim)
            state['best_f'] = call(state['best_x'])

        return state['best_x'], state['best_f']
# EVOLVE-BLOCK-END
