"""Surrogate-assisted optimizer for BBOB-style f15 (Rastrigin rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import warnings

class Optimizer:
    """Matern GP surrogate with Expected Improvement acquisition."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim
        self.rng = np.random.default_rng(42)
        # Normalize inputs to [0, 1] for GP stability
        self.x_min = -5.0
        self.x_max = 5.0
        self.x_range = self.x_max - self.x_min

    def _matern52_kernel(self, X1, X2, length_scales, sigma_f):
        """
        Compute Matern 5/2 kernel with ARD.
        X1, X2: (n, dim) arrays normalized to [0, 1]
        length_scales: (dim,) array
        """
        # Compute squared distances scaled by length scales
        # X1: (n1, d), X2: (n2, d)
        # dist^2 = sum_i ( (x1_i - x2_i)^2 / l_i^2 )
        
        # Efficient computation
        # X1_scaled = X1 / length_scales
        # X2_scaled = X2 / length_scales
        # sq_dist = sum(X1_scaled^2, axis=1)[:, None] + sum(X2_scaled^2, axis=1)[None, :] - 2 * X1_scaled @ X2_scaled.T
        
        X1_scaled = X1 / length_scales
        X2_scaled = X2 / length_scales
        
        sq_dist = np.sum(X1_scaled**2, axis=1)[:, np.newaxis] + \
                  np.sum(X2_scaled**2, axis=1)[np.newaxis, :] - \
                  2.0 * (X1_scaled @ X2_scaled.T)
        
        sq_dist = np.maximum(sq_dist, 0.0)
        r = np.sqrt(sq_dist)
        
        # Matern 5/2
        sqrt3_r = np.sqrt(3.0) * r
        K = sigma_f**2 * (1.0 + sqrt3_r + 0.5 * sqrt3_r**2) * np.exp(-sqrt3_r)
        
        return K

    def _fit_gp(self, X_train, y_train):
        """
        Fit GP. X_train, y_train are original scale.
        Returns predict function.
        """
        n = len(X_train)
        if n == 0:
            return lambda X: (np.zeros(len(X)), np.ones(len(X)))
        
        # Normalize inputs
        X_train_norm = (X_train - self.x_min) / self.x_range
        
        # Hyperparameters
        # Initialize length scales to 1.0 (in normalized space, so full range is 1 unit)
        # This is a reasonable default.
        length_scales = np.ones(self.dim)
        sigma_f = 1.0
        sigma_n = 1e-6 # Small noise for numerical stability
        
        # Scale y to [0, 1] or center it? 
        # GP is linear in y. It's often better to center y.
        y_mean = np.mean(y_train)
        y_std = np.std(y_train) + 1e-8
        y_norm = (y_train - y_mean) / y_std
        
        K = self._matern52_kernel(X_train_norm, X_train_norm, length_scales, sigma_f)
        K[np.diag_indices(n)] += sigma_n**2
        
        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
        except np.linalg.LinAlgError:
            # Add jitter
            K[np.diag_indices(n)] += 1e-4
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
            except np.linalg.LinAlgError:
                # Fallback
                mean_y = y_mean
                var_y = y_std**2 + 1.0
                def predict_fallback(X):
                    return np.full(len(X), mean_y), np.full(len(X), np.sqrt(var_y))
                return predict_fallback

        def predict(X):
            X_norm = (X - self.x_min) / self.x_range
            K_star = self._matern52_kernel(X_norm, X_train_norm, length_scales, sigma_f)
            mu_norm = K_star @ alpha
            # Unnormalize mean
            mu = mu_norm * y_std + y_mean
            
            v = np.linalg.solve(L, K_star.T)
            var_norm = sigma_f**2 - np.sum(v**2, axis=1)
            var_norm = np.maximum(var_norm, 1e-8)
            # Unnormalize variance
            var = var_norm * y_std**2
            std = np.sqrt(var)
            
            return mu, std
        
        return predict

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = self.rng
        
        X_list = []
        y_list = []
        
        # Initial points: Latin Hypercube Sampling for better coverage
        n_init = 10
        # Generate LHS points in [0, 1]^dim
        lhs_points = np.zeros((n_init, dim))
        for i in range(dim):
            points = np.linspace(0, 1, n_init, endpoint=False)
            rng.shuffle(points)
            lhs_points[:, i] = points
        
        # Map to [-5, 5]
        lhs_points_orig = lhs_points * self.x_range + self.x_min
        
        for x in lhs_points_orig:
            y = func(x)
            X_list.append(x)
            y_list.append(y)
            
        X = np.array(X_list)
        y = np.array(y_list)
        
        best_x = X[np.argmin(y)]
        best_y = min(y)
        
        calls_made = len(X_list)
        
        while calls_made < budget:
            # Fit GP
            predict_func = self._fit_gp(X, y)
            
            # Define acquisition function (Expected Improvement)
            def acquisition(x):
                x = np.array(x).reshape(1, -1)
                mu, std = predict_func(x)
                mu = mu[0]
                std = std[0]
                
                if std < 1e-10:
                    return best_y - mu # Exploit if certain
                
                z = (best_y - mu) / std
                ei = (best_y - mu) * norm.cdf(z) + std * norm.pdf(z)
                return ei
            
            # Maximize acquisition function using multiple restarts
            best_acq_val = -np.inf
            best_x_acq = None
            
            # Use points from current data and random points as restarts
            restarts = []
            # Add best point
            restarts.append(best_x)
            # Add some random points
            for _ in range(10):
                restarts.append(rng.uniform(self.x_min, self.x_max, dim))
            # Add some points from existing data (shuffled)
            idxs = rng.choice(len(X), size=min(10, len(X)), replace=False)
            for i in idxs:
                restarts.append(X[i])
                
            for r in restarts:
                # L-BFGS-B to maximize acquisition
                res = minimize(
                    lambda x: -acquisition(x), 
                    r, 
                    method='L-BFGS-B', 
                    bounds=[(self.x_min, self.x_max)] * dim
                )
                if res.success or res.fun < -best_acq_val:
                    acq_val = -res.fun
                    if acq_val > best_acq_val:
                        best_acq_val = acq_val
                        best_x_acq = res.x
            
            if best_x_acq is None:
                # Fallback to random
                best_x_acq = rng.uniform(self.x_min, self.x_max, dim)
            
            # Evaluate
            y_new = func(best_x_acq)
            X_list.append(best_x_acq)
            y_list.append(y_new)
            
            X = np.vstack([X, best_x_acq.reshape(1, -1)])
            y = np.append(y, y_new)
            calls_made += 1
            
            if y_new < best_y:
                best_y = y_new
                best_x = best_x_acq
                
        return best_x, best_y
# EVOLVE-BLOCK-END
