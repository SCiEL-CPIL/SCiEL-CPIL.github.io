import numpy as np
import time
from scipy.stats import norm
from scipy.optimize import minimize

class Optimizer:
    """Bayesian Optimization with Gaussian Process for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(123)

        calls_made = 0
        best_x = None
        best_f = np.inf

        # Phase 1: Initial sampling using Sobol/Latin Hypercube for better coverage
        n_init = 40
        # Simple Latin Hypercube sampling
        X = np.zeros((n_init, dim))
        for d in range(dim):
            X[:, d] = rng.permutation(n_init) * (10.0 / n_init) - 5.0 + rng.uniform(0, 10.0/n_init, n_init)
            X[:, d] = np.clip(X[:, d], -5, 5)
            
        Y = np.array([func(np.clip(x, -5, 5)) for x in X])
        calls_made += n_init

        best_idx = np.argmin(Y)
        best_f = Y[best_idx]
        best_x = X[best_idx].copy()

        # GP Hyperparameters
        length_scale = 1.5
        variance = 10.0
        noise = 1e-6

        def matern52_kernel(X1, X2, length_scale, variance):
            dist = np.sqrt(np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T)
            dist = np.maximum(dist, 1e-10)
            sqrt3_dist = np.sqrt(3) * dist / length_scale
            K = variance * (1 + sqrt3_dist) * np.exp(-sqrt3_dist)
            return K

        def compute_gp_posterior(X_obs, Y_obs, X_new, length_scale, variance, noise):
            K = matern52_kernel(X_obs, X_obs, length_scale, variance) + noise * np.eye(len(X_obs))
            Ks = matern52_kernel(X_obs, X_new, length_scale, variance)
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K += 1e-8 * np.eye(len(K))
                L = np.linalg.cholesky(K)
            
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_obs))
            mu = Ks.T @ alpha
            v = np.linalg.solve(L, Ks)
            cov = variance - v.T @ v
            var = np.diag(cov)
            var = np.maximum(var, 1e-10)
            return mu, var

        def expected_improvement(x, X_obs, Y_obs, best_f, length_scale, variance, noise, xi=0.01):
            x = np.array([x])
            mu, var = compute_gp_posterior(X_obs, Y_obs, x, length_scale, variance, noise)
            std = np.sqrt(var[0])
            if std < 1e-8:
                return 0.0
            z = (best_f - mu[0] - xi) / std
            ei = (best_f - mu[0] - xi) * norm.cdf(z) + std * norm.pdf(z)
            return -ei  # Minimize negative EI

        n_bo = budget - n_init
        for step in range(n_bo):
            if time.time() > deadline:
                break
            if calls_made >= budget:
                break

            # Adaptive length scale
            if best_f < 1.0:
                current_scale = 0.5
            elif best_f < 10.0:
                current_scale = 1.0
            else:
                current_scale = 2.0

            # Optimize acquisition function using multi-start Nelder-Mead
            # We use multiple starting points to avoid local maxima of EI
            n_starts = 5
            best_ei_val = -np.inf
            best_x_next = None
            
            # Start points: best_x, random points
            starts = [best_x]
            for _ in range(n_starts - 1):
                starts.append(rng.uniform(-5, 5, size=dim))
                
            for start in starts:
                res = minimize(
                    expected_improvement,
                    start,
                    args=(X, Y, best_f, current_scale, variance, noise),
                    method='Nelder-Mead',
                    options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 100}
                )
                if res.fun < best_ei_val:
                    best_ei_val = res.fun
                    best_x_next = res.x
            
            if best_x_next is None:
                break
                
            x_next = np.clip(best_x_next, -5, 5)
            f_next = func(x_next)
            calls_made += 1
            
            X = np.vstack([X, x_next])
            Y = np.append(Y, f_next)
            
            if f_next < best_f:
                best_f = f_next
                best_x = x_next.copy()

        return best_f
