"""Surrogate-assisted optimizer using Gaussian Process + Expected Improvement."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from scipy.stats.qmc import Sobol

class Optimizer:
    """GP surrogate with Expected Improvement acquisition."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        # 1. Initialization with Sobol sequence for better space-filling
        n_init = min(20, budget)
        sampler = Sobol(d=dim, scramble=True)
        X = sampler.random(n_init) * 10 - 5  # Scale to [-5, 5]
        Y = np.array([func(x) for x in X])
        
        best_idx = np.argmin(Y)
        x_best = X[best_idx].copy()
        f_best = Y[best_idx]
        
        # Normalize Y using initial statistics for stability
        y_mean = np.mean(Y)
        y_std = np.std(Y)
        if y_std < 1e-8:
            y_std = 1.0
        Y_norm = (Y - y_mean) / y_std
        
        # GP Hyperparameters
        # Matern 2.5 kernel is often better for smooth but not infinitely differentiable functions
        length_scale = 1.5
        signal_var = 1.0
        noise_var = 1e-4
        
        def kernel(A, B):
            """Matern 2.5 kernel."""
            diff = A[:, None, :] - B[None, :, :]
            sq_dist = np.sum(diff ** 2, axis=-1)
            dist = np.sqrt(sq_dist)
            # Matern nu=2.5
            return signal_var * (1 + np.sqrt(5) * dist / length_scale) * np.exp(-np.sqrt(5) * dist / length_scale)

        def gp_predict(X_new, X_train, Y_train_norm):
            """Predict mean and variance at X_new given training data."""
            K = kernel(X_train, X_train)
            K += noise_var * np.eye(len(X_train))
            K_new = kernel(X_train, X_new)
            
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train_norm))
            except np.linalg.LinAlgError:
                K += 1e-6 * np.eye(len(X_train))
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train_norm))
            
            mean_norm = K_new.T @ alpha
            v = np.linalg.solve(L, K_new.T)
            var_norm = signal_var - np.sum(v ** 2, axis=0)
            var_norm = np.maximum(var_norm, 1e-8)
            
            # Denormalize mean and variance
            mean = mean_norm * y_std + y_mean
            var = var_norm * (y_std ** 2)
            
            return mean, var

        def expected_improvement(x, f_best, X_train, Y_train_norm):
            """Compute EI for a single point x."""
            x = np.array([x])
            mean, var = gp_predict(x, X_train, Y_train_norm)
            sigma = np.sqrt(var[0])
            if sigma < 1e-8:
                return 0.0
            z = (f_best - mean[0]) / sigma
            ei = (f_best - mean[0]) * norm.cdf(z) + sigma * norm.pdf(z)
            return -ei  # Negative for minimization

        # Main optimization loop
        calls_made = n_init
        X_train = X
        Y_train_norm = Y_norm
        
        for _ in range(budget - n_init):
            # Find next point using EI
            # Use Nelder-Mead for derivative-free optimization of EI
            n_restarts = 10
            best_ei = -np.inf
            x_next = None
            
            # Start points: current best, centroid, and random
            centroid = np.mean(X_train, axis=0)
            start_points = [x_best, centroid] + [rng.uniform(-5, 5, dim) for _ in range(n_restarts - 2)]
            
            for x0 in start_points:
                x0 = np.clip(x0, -5, 5)
                try:
                    res = minimize(
                        lambda x: expected_improvement(x, f_best, X_train, Y_train_norm),
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6}
                    )
                    if res.fun < best_ei:
                        best_ei = res.fun
                        x_next = np.clip(res.x, -5, 5)
                except Exception:
                    continue
            
            if x_next is None:
                x_next = rng.uniform(-5, 5, dim)
            
            # Evaluate the new point
            f_new = func(x_next)
            
            # Update training data
            X_train = np.vstack([X_train, x_next])
            
            # Update Y_train_norm using initial normalization
            Y_train_norm = np.append(Y_train_norm, (f_new - y_mean) / y_std)
            
            calls_made += 1
            
            # Update best
            if f_new < f_best:
                f_best = f_new
                x_best = x_next.copy()
        
        return x_best, f_best
# EVOLVE-BLOCK-END
