"""Bayesian Optimization with Gaussian Process surrogate for BBOB f4."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import warnings


class Optimizer:
    """Bayesian Optimization with GP surrogate for 5D Büche-Rastrigin."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _length_scale(self, n_points):
        """Adaptive length scale based on number of observations."""
        # Start with relatively large length scale, decrease as we learn
        return max(0.5, 3.0 / (1 + 0.1 * n_points))

    def _acquire(self, X, y, l, sigma_f, xi):
        """Expected Improvement acquisition function."""
        y_mean = y.mean()
        y_std = max(y.std(), 1e-8)
        y = (y - y_mean) / y_std  # normalize for stability
        
        X = np.atleast_2d(X)
        n = X.shape[0]
        d = X.shape[1]
        
        # RBF kernel
        # Compute pairwise distances
        sq_dists = np.sum(X**2, axis=1)[:, None] + np.sum(X**2, axis=1)[None, :] - 2 * X @ X.T
        sq_dists = np.maximum(sq_dists, 0)
        K = np.exp(-sq_dists / (2 * l**2))
        
        # Add noise to diagonal
        K += 1e-8 * np.eye(n)
        
        # Predict mean and variance at candidate points
        # For efficiency, we evaluate EI on a grid
        return K, y

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # Initial points: mix of random and structured
        X = np.zeros((0, dim))
        y = np.zeros(0)
        
        def evaluate(x):
            x = np.clip(x, -5, 5)
            return func(x)
        
        # Phase 1: Initial exploration with random points
        n_init = min(20, budget)
        for i in range(n_init):
            if i < dim:  # Axis-aligned exploration
                x = np.zeros(dim)
                x[i] = rng.uniform(-4, 4)
            elif i < 2 * dim:  # More random
                x = rng.uniform(-4, 4, dim)
            else:
                x = rng.uniform(-5, 5, dim)
            x = np.clip(x, -5, 5)
            fx = evaluate(x)
            X = np.vstack([X, x.reshape(1, -1)]) if len(X) > 0 else x.reshape(1, -1)
            y = np.append(y, fx)
        
        if len(X) == 0:
            # Shouldn't happen but safety
            x0 = rng.uniform(-5, 5, dim)
            fx = evaluate(x0)
            return x0, fx
        
        best_x = X[np.argmin(y)]
        best_y = y.min()
        
        calls_made = n_init
        remaining = budget - calls_made
        
        # GP hyperparameters
        l = 1.5  # length scale
        sigma_f = 1.0  # signal std (will be normalized)
        xi = 0.01  # exploration parameter
        
        # Phase 2: Bayesian Optimization loop
        for iteration in range(remaining):
            if len(X) < 2:
                # Need at least 2 points for GP
                x_new = rng.uniform(-5, 5, dim)
                fx = evaluate(x_new)
                X = np.vstack([X, x_new.reshape(1, -1)])
                y = np.append(y, fx)
                calls_made += 1
                if fx < best_y:
                    best_y = fx
                    best_x = x_new
                continue
            
            # Normalize y for numerical stability
            y_mean = y.mean()
            y_std = max(y.std(), 1e-8)
            y_norm = (y - y_mean) / y_std
            
            n = len(X)
            
            # Compute GP posterior
            # RBF kernel
            sq_dists = np.sum(X**2, axis=1)[:, None] + np.sum(X**2, axis=1)[None, :] - 2 * X @ X.T
            sq_dists = np.maximum(sq_dists, 0)
            K = np.exp(-sq_dists / (2 * l**2))
            K += 1e-6 * np.eye(n)
            
            # Solve K alpha = y
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
            except np.linalg.LinAlgError:
                # Fallback to least squares
                try:
                    alpha = np.linalg.lstsq(K, y_norm, rcond=None)[0]
                except:
                    alpha = np.zeros(n)
            
            # Define acquisition function (Expected Improvement)
            def ei(x_candidate):
                x_candidate = x_candidate.reshape(1, -1)
                # Compute kernel between x_candidate and training points
                sq_dist = np.sum((X - x_candidate)**2, axis=1)
                k = np.exp(-sq_dist / (2 * l**2))
                
                # Predictive mean and variance
                mu = k @ alpha
                v = np.linalg.solve(L, k)
                var = 1.0 - k @ v
                var = max(var, 1e-10)
                std = np.sqrt(var)
                
                # Expected Improvement
                if std < 1e-8:
                    return 0.0
                z = (mu - y_norm.min()) / std
                ei_val = (mu - y_norm.min()) * norm.cdf(z) + std * norm.pdf(z)
                return -ei_val  # Negative for minimization
            
            # Optimize the acquisition function using multiple random restarts
            best_x_acq = None
            best_ei = np.inf
            
            # Try several random starting points
            for _ in range(10):
                x0 = rng.uniform(-5, 5, dim)
                try:
                    res = minimize(ei, x0, method='L-BFGS-B', bounds=[(-5, 5)] * dim,
                                   options={'maxiter': 100, 'ftol': 1e-8})
                    if res.success or res.fun < best_ei:
                        best_ei = res.fun
                        best_x_acq = res.x
                except:
                    pass
            
            # Also try perturbing the current best
            for _ in range(5):
                x0 = best_x + rng.standard_normal(dim) * 0.5
                x0 = np.clip(x0, -5, 5)
                try:
                    res = minimize(ei, x0, method='L-BFGS-B', bounds=[(-5, 5)] * dim,
                                   options={'maxiter': 100, 'ftol': 1e-8})
                    if res.success or res.fun < best_ei:
                        best_ei = res.fun
                        best_x_acq = res.x
                except:
                    pass
            
            if best_x_acq is None:
                # Fallback: random point
                best_x_acq = rng.uniform(-5, 5, dim)
            
            best_x_acq = np.clip(best_x_acq, -5, 5)
            
            # Evaluate the selected point
            fx = evaluate(best_x_acq)
            X = np.vstack([X, best_x_acq.reshape(1, -1)])
            y = np.append(y, fx)
            calls_made += 1
            
            if fx < best_y:
                best_y = fx
                best_x = best_x_acq
            
            # Adaptively adjust exploration/exploitation
            # Increase exploration early, decrease later
            progress = calls_made / budget
            if progress < 0.3:
                l = 2.0
                xi = 0.05
            elif progress < 0.7:
                l = 1.0
                xi = 0.02
            else:
                l = 0.5
                xi = 0.01
            
            # Occasionally inject random points to avoid local optima
            if rng.random() < 0.05 and calls_made < budget:
                x_rand = rng.uniform(-5, 5, dim)
                fx_rand = evaluate(x_rand)
                X = np.vstack([X, x_rand.reshape(1, -1)])
                y = np.append(y, fx_rand)
                calls_made += 1
                if fx_rand < best_y:
                    best_y = fx_rand
                    best_x = x_rand
                # Break the loop if we've used up budget
                if calls_made >= budget:
                    break
        
        return best_x, best_y
# EVOLVE-BLOCK-END
