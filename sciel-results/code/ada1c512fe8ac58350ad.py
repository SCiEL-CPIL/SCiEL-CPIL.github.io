"""Bayesian Optimization with GP surrogate for BBOB f14."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


class Optimizer:
    """Bayesian Optimizer using GP surrogate with Expected Improvement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _latin_hypercube(self, n, dim, rng):
        """Generate n points via Latin Hypercube sampling in [-5, 5]^dim."""
        points = np.zeros((n, dim))
        for d in range(dim):
            perm = rng.permutation(n)
            points[:, d] = (perm + 0.5) / n * 10 - 5
        return points

    def _matern52_kernel(self, X1, X2, length_scales, variance):
        """Matern 5/2 kernel."""
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        d = X1.shape[1]
        
        # Compute squared distances
        sq_dists = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
        sq_dists = np.maximum(sq_dists, 0)
        
        # Scale by length scales
        ls = length_scales
        scaled_dists = sq_dists / (ls**2)
        
        r = np.sqrt(scaled_dists)
        
        # Matern 5/2 kernel
        K = variance * (1 + np.sqrt(5) * r + 5/3 * r**2) * np.exp(-np.sqrt(5) * r)
        
        # Add small jitter for numerical stability
        K[np.diag_indices_from(K)] += 1e-10
        return K

    def _fit_gp_params(self, X, y, rng):
        """Fit GP hyperparameters by maximizing log marginal likelihood."""
        d = X.shape[1]
        n = X.shape[0]
        
        # Initial guesses
        length_scales = np.ones(d) * 1.0
        variance = 1.0
        noise = 1e-6
        
        # Simple coordinate descent / grid search for hyperparameters
        best_ll = -np.inf
        best_params = (length_scales.copy(), variance, noise)
        
        # Try a few different initializations
        for ls_init in [0.5, 1.0, 2.0]:
            for var_init in [0.5, 1.0, 5.0]:
                ls = np.ones(d) * ls_init
                var = var_init
                
                # Optimize length scales and variance
                def neg_log_marginal(params):
                    ls_vals = np.abs(params[:d]) + 1e-4
                    var_val = np.abs(params[d]) + 1e-4
                    noise_val = np.abs(params[d+1]) + 1e-8
                    
                    K = self._matern52_kernel(X, X, ls_vals, var_val)
                    K += noise_val * np.eye(n)
                    
                    try:
                        L = np.linalg.cholesky(K)
                    except np.linalg.LinAlgError:
                        return 1e10
                    
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                    ll = -0.5 * y @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * n * np.log(2 * np.pi)
                    return -ll
                
                x0 = np.concatenate([ls, [var, noise]])
                res = minimize(neg_log_marginal, x0, method='Nelder-Mead', 
                              options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-4})
                
                if res.fun < best_ll:
                    best_ll = res.fun
                    best_params = (np.abs(res.x[:d]) + 1e-4, np.abs(res.x[d]) + 1e-4, np.abs(res.x[d+1]) + 1e-8)
        
        return best_params

    def _gp_predict(self, X_new, X_train, y_train, length_scales, variance, noise):
        """Predict mean and variance at X_new using GP."""
        n_train = X_train.shape[0]
        n_new = X_new.shape[0]
        
        K_train = self._matern52_kernel(X_train, X_train, length_scales, variance)
        K_train += noise * np.eye(n_train)
        K_new = self._matern52_kernel(X_new, X_train, length_scales, variance)
        
        L = np.linalg.cholesky(K_train)
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        
        mu = K_new @ alpha
        v = np.linalg.solve(L, K_new.T)
        var = variance - np.sum(v**2, axis=0)
        var = np.maximum(var, 1e-12)
        
        return mu, var

    def _expected_improvement(self, mu, var, y_best, xi=0.01):
        """Compute Expected Improvement."""
        sigma = np.sqrt(var)
        z = (y_best - mu - xi) / (sigma + 1e-10)
        from scipy.stats import norm
        ei = (y_best - mu - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei

    def _local_polish(self, func, x0, remaining_budget, deadline):
        """Polish using Nelder-Mead, respecting budget and deadline."""
        best_x = x0.copy()
        best_f = func(x0)
        calls_used = 1
        
        while calls_used < remaining_budget and time.time() < deadline:
            res = minimize(func, best_x, method='Nelder-Mead', 
                          options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8})
            calls_used += res.nfev
            if res.fun < best_f:
                best_x = res.x.copy()
                best_f = res.fun
            else:
                break
        
        return best_x, best_f, calls_used

    def __call__(self, func):
        rng = np.random.default_rng(42)
        deadline = time.time() + 290  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Latin Hypercube initialization
        n_init = min(30, budget // 5)
        X_init = self._latin_hypercube(n_init, dim, rng)
        y_init = np.array([func(x) for x in X_init])
        
        X = X_init.copy()
        y = y_init.copy()
        calls_made = n_init
        
        # Find best
        best_idx = np.argmin(y)
        best_x = X[best_idx].copy()
        best_f = y[best_idx]
        
        # Phase 2: Bayesian Optimization
        # Fit GP parameters once initially, then periodically
        length_scales, variance, noise = self._fit_gp_params(X, y, rng)
        
        gp_refit_interval = 10
        since_refit = 0
        
        while calls_made < budget and time.time() < deadline:
            # Predict on a candidate set
            n_candidates = 50
            X_cand = rng.uniform(-5, 5, (n_candidates, dim))
            
            mu, var = self._gp_predict(X_cand, X, y, length_scales, variance, noise)
            
            # Expected Improvement
            ei = self._expected_improvement(mu, var, best_f)
            
            # Select point with max EI
            best_cand_idx = np.argmax(ei)
            x_next = X_cand[best_cand_idx].copy()
            
            # Evaluate
            f_next = func(x_next)
            calls_made += 1
            
            # Update
            X = np.vstack([X, x_next])
            y = np.append(y, f_next)
            
            if f_next < best_f:
                best_f = f_next
                best_x = x_next.copy()
            
            since_refit += 1
            if since_refit >= gp_refit_interval:
                length_scales, variance, noise = self._fit_gp_params(X, y, rng)
                since_refit = 0
        
        # Phase 3: Local polish with remaining budget
        remaining = budget - calls_made
        if remaining > 0 and time.time() < deadline:
            best_x, best_f, polish_calls = self._local_polish(func, best_x, remaining, deadline)
            calls_made += polish_calls
        
        return best_x, best_f
# EVOLVE-BLOCK-END
