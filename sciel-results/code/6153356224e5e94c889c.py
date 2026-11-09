"""Hybrid Bayesian Optimization (GP) with (1+1)-ES Polish for BBOB f23 (Katsuura)."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm


class Optimizer:
    """
    Hybrid optimizer:
    1. Gaussian Process (Matern 5/2) with EI for global search.
    2. (1+1)-ES for local high-precision polish.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = np.full(dim, -5.0)
        self.ub = np.full(dim, 5.0)

    def _latin_hypercube(self, n, dim, rng):
        pts = np.empty((n, dim))
        for d in range(dim):
            order = rng.permutation(n)
            pts[order, d] = (np.arange(n) + rng.random(n)) / n
        return self.lb + pts * (self.ub - self.lb)

    def _matern52_kernel(self, X, X_train, length_scale):
        # Compute Euclidean distances
        # d2 = sum((x - x')^2)
        d2 = np.sum(X**2, axis=1)[:, None] - 2 * X @ X_train.T + np.sum(X_train**2, axis=1)[None, :]
        d2 = np.maximum(d2, 0) # numerical stability
        d = np.sqrt(d2)
        
        # Matern 5/2
        sqrt5 = np.sqrt(5.0)
        K = (1 + sqrt5 * d + (5.0 / 3.0) * d**2) * np.exp(-sqrt5 * d)
        return K

    def _gp_predict(self, X, X_train, y_train, length_scale, sigma_f, noise):
        K = self._matern52_kernel(X_train, X_train, length_scale)
        Kss = K + (noise + sigma_f**2) * np.eye(len(X_train)) # Standard GP noise term is usually just noise, but let's be robust
        # Actually, standard form: K + sigma_noise^2 I. sigma_f is signal variance.
        # Let's use K + noise * I where noise includes measurement noise.
        # To be safe: K + (sigma_f**2 * 1e-6 + noise) * I? 
        # Let's stick to K + noise * I and assume y is standardized or sigma_f is scaled.
        # Better: Standardize y in GP? No, keep raw.
        # Kss = K + noise * np.eye(len(X_train))
        
        # Add jitter for stability
        Kss = K + noise * np.eye(len(X_train))
        Kss += 1e-8 * np.eye(len(X_train))
        
        try:
            L = np.linalg.cholesky(Kss)
        except np.linalg.LinAlgError:
            Kss += 1e-6 * np.eye(len(X_train))
            L = np.linalg.cholesky(Kss)
            
        # Solve for alpha
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        
        # Predict mean and variance for new points X
        K_star = self._matern52_kernel(X, X_train, length_scale)
        mu = K_star @ alpha
        
        v = np.linalg.solve(L, K_star.T)
        var = sigma_f**2 - np.sum(v**2, axis=0)
        var = np.maximum(var, 1e-12)
        
        return mu, np.sqrt(var)

    def _ei(self, mu, sigma, best_f, xi=0.05):
        z = (best_f - xi - mu) / (sigma + 1e-12)
        ei = (best_f - xi - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        dim = self.dim
        lb, ub = self.lb, self.ub
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        # --- Phase 1: Global Search with GP ---
        
        # Initial Latin Hypercube
        n_init = 40
        X = self._latin_hypercube(n_init, dim, rng)
        y = np.array([func(x) for x in X])
        
        calls_made = n_init
        idx_best = np.argmin(y)
        x_best = X[idx_best].copy()
        f_best = y[idx_best]
        
        # GP Hyperparameters
        length_scale = 1.0
        sigma_f = 1.0
        noise = 1e-5
        
        # Number of steps for GP phase
        # Reserve at least 100 calls for polish
        n_gp_steps = min(budget - calls_made, 250)
        
        for i in range(n_gp_steps):
            if time.time() > deadline - 30.0:
                break
            if calls_made >= budget:
                break
                
            # Generate candidates
            # 1. Random points
            n_rand = 100
            X_rand = rng.uniform(lb, ub, (n_rand, dim))
            
            # 2. Points near current best
            n_near = 50
            X_near = x_best + rng.normal(0, 0.5, (n_near, dim))
            X_near = np.clip(X_near, lb, ub)
            
            X_cand = np.vstack([X_rand, X_near])
            
            # Predict
            mu, sigma = self._gp_predict(X_cand, X, y, length_scale, sigma_f, noise)
            
            # EI
            ei_vals = self._ei(mu, sigma, f_best, xi=0.05)
            
            # Select best
            idx = np.argmax(ei_vals)
            x_new = X_cand[idx].copy()
            
            # Evaluate
            f_new = func(x_new)
            calls_made += 1
            
            # Update
            X = np.vstack([X, x_new[None, :]])
            y = np.append(y, f_new)
            
            if f_new < f_best:
                f_best = f_new
                x_best = x_new.copy()
                
            # Adaptive hyperparameters
            # Shrink length scale slowly to focus on local structure
            if i % 20 == 0 and i > 0:
                length_scale = max(length_scale * 0.95, 0.2)
                
            # Update sigma_f to track scale
            if i % 10 == 0:
                sigma_f = max(np.std(y), 0.1)

        # --- Phase 2: Local Polish with (1+1)-ES ---
        
        if calls_made < budget and time.time() < deadline - 5.0:
            # Initialize ES parameters
            # Start with a small sigma, assuming GP brought us close
            sigma = 0.2 
            x_es = x_best.copy()
            fx_es = f_best
            
            # If GP didn't improve much, maybe start sigma larger?
            # But 0.2 is reasonable for 5D box [-5,5]
            
            while calls_made < budget:
                if time.time() > deadline - 2.0:
                    break
                    
                # Generate candidate
                y_es = x_es + sigma * rng.standard_normal(dim)
                y_es = np.clip(y_es, lb, ub)
                
                fy_es = func(y_es)
                calls_made += 1
                
                if fy_es < fx_es:
                    x_es, fx_es = y_es, fy_es
                    sigma *= 1.2
                    if fx_es < f_best:
                        f_best = fx_es
                        x_best = x_es.copy()
                else:
                    sigma *= 0.9
                    
                # Prevent sigma from becoming too large or small
                sigma = np.clip(sigma, 1e-8, 5.0)

        return x_best, f_best
# EVOLVE-BLOCK-END
