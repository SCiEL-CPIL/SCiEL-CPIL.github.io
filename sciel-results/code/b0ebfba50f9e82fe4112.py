"""Bayesian Optimization surrogate (Gaussian Process) with Expected Improvement for BBOB f8."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """Gaussian Process Bayesian Optimizer with Expected Improvement acquisition."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _gp_predict(self, X_new, X, y, length_scale, noise):
        """
        Compute GP mean and variance at X_new given training data X, y.
        Uses a simple RBF kernel.
        """
        n = X.shape[0]
        m = X_new.shape[0]
        
        # Compute kernel matrix K(X, X)
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        sq_dist = np.sum(diff ** 2, axis=-1)
        K = np.exp(-0.5 * sq_dist / (length_scale ** 2)) + noise * np.eye(n)
        
        # Precompute Cholesky decomposition
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # Add jitter if Cholesky fails
            K += 1e-8 * np.eye(n)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # Fallback to eigendecomposition
                eigvals, eigvecs = np.linalg.eigh(K)
                eigvals = np.maximum(eigvals, 1e-10)
                L = eigvecs @ np.diag(np.sqrt(eigvals))
        
        # Compute alpha = K^{-1} y
        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
        
        # Compute K(X_new, X)
        diff_new = X_new[:, np.newaxis, :] - X[np.newaxis, :, :]
        sq_dist_new = np.sum(diff_new ** 2, axis=-1)
        K_star = np.exp(-0.5 * sq_dist_new / (length_scale ** 2))
        
        # Predict mean
        mean = K_star @ alpha
        
        # Predict variance
        v = np.linalg.solve(L, K_star.T)
        var = 1.0 - np.sum(v ** 2, axis=0)
        var = np.maximum(var, 1e-10)
        
        return mean, var

    def _expected_improvement(self, x, y_best, mean, var, xi=0.01):
        """Compute Expected Improvement acquisition function."""
        sigma = np.sqrt(var)
        z = (y_best - mean - xi) / sigma
        ei = (y_best - mean - xi) * norm.cdf(z) + sigma * norm.pdf(z)
        return ei

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Initial points: random sampling + center
        n_init = min(10, budget)
        X = []
        y = []
        
        # Add center
        X.append(np.zeros(dim))
        y.append(func(np.zeros(dim)))
        
        # Add random points
        for _ in range(n_init - 1):
            x = rng.uniform(-5, 5, dim)
            X.append(x)
            y.append(func(x))
        
        X = np.array(X)
        y = np.array(y)
        
        # Adaptive length scale and noise
        length_scale = 1.5
        noise = 1e-6
        
        # Track best
        y_best = np.min(y)
        x_best = X[np.argmin(y)]
        
        # Bayesian optimization loop
        for i in range(n_init, budget):
            # If we've been stuck, try to adapt
            if i > n_init + 50:
                # Slightly reduce length scale to focus on local
                length_scale = max(0.5, length_scale * 0.99)
            
            # Generate candidate points for acquisition
            # Use a mix of random points and local perturbations
            n_candidates = 200
            
            # Random candidates
            candidates = rng.uniform(-5, 5, (n_candidates, dim))
            
            # Add local perturbations around best point
            if i > n_init + 20:
                n_local = 50
                local_sigma = max(0.5, length_scale * 0.5)
                local_candidates = x_best + local_sigma * rng.standard_normal((n_local, dim))
                local_candidates = np.clip(local_candidates, -5, 5)
                candidates = np.vstack([candidates, local_candidates])
            
            # Evaluate acquisition function
            mean, var = self._gp_predict(candidates, X, y, length_scale, noise)
            ei = self._expected_improvement(candidates, y_best, mean, var)
            
            # Select best candidate
            best_idx = np.argmax(ei)
            x_new = candidates[best_idx]
            
            # Evaluate function
            f_new = func(x_new)
            
            # Update data
            X = np.vstack([X, x_new])
            y = np.append(y, f_new)
            
            # Update best
            if f_new < y_best:
                y_best = f_new
                x_best = x_new
                # Increase length scale slightly when we find improvement
                length_scale = min(3.0, length_scale * 1.1)
        
        return x_best, y_best
# EVOLVE-BLOCK-END
