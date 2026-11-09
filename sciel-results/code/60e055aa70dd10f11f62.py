import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.start_time = None
        self.max_time = 290.0  # Safety margin for 300s limit

    def _halton_sequence(self, n, dim):
        """Generate n points in [0,1]^dim using Halton sequence."""
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        points = np.zeros((n, dim))
        for i in range(n):
            for j in range(dim):
                # Use the first 'dim' prime numbers
                primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
                points[i, j] = halton(i + 1, primes[j])
        return points

    def _kernel(self, X1, X2, length_scale, variance):
        """RBF Kernel."""
        sq_dists = np.sum(X1**2, 1)[:, None] - 2 * np.dot(X1, X2.T) + np.sum(X2**2, 1)[None, :]
        sq_dists = np.maximum(sq_dists, 0)
        return variance * np.exp(-0.5 * sq_dists / (length_scale ** 2))

    def _gp_predict(self, X_new, X_train, y_train, length_scale, variance, noise):
        """Predict mean and variance using GP."""
        n = len(X_train)
        K = self._kernel(X_train, X_train, length_scale, variance) + noise * np.eye(n)
        # Add jitter for numerical stability
        K += 1e-8 * np.eye(n)
        
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            # Fallback to inv if cholesky fails
            K_inv = np.linalg.inv(K)
            K_star = self._kernel(X_new, X_train, length_scale, variance)
            mu = K_star @ K_inv @ y_train
            # Variance calculation is more complex with inv, approximating with diagonal
            v = np.linalg.solve(L, y_train) if 'L' in locals() else K_inv @ y_train
            # Simpler fallback: just use mean, high variance
            return mu, np.ones(len(X_new)) * variance

        K_star = self._kernel(X_new, X_train, length_scale, variance)
        
        # Solve K v = y
        v = np.linalg.solve(L, y_train)
        # mu = K_star v
        mu = K_star @ v
        # var = variance - K_star K^{-1} K_star.T
        # var = variance - sum(K_star * K_star.T * (K^{-1})) -> use cholesky
        # v_star = L^{-1} K_star.T
        v_star = np.linalg.solve(L, K_star.T)
        var = variance - np.sum(v_star * K_star, axis=1)
        var = np.maximum(var, 1e-10)
        
        return mu, var

    def _ei(self, x, f_min, mu, var, length_scale, variance, noise, X_train, y_train):
        """Calculate Expected Improvement for a single point x."""
        # x is shape (dim,)
        x_new = x.reshape(1, -1)
        m, v = self._gp_predict(x_new, X_train, y_train, length_scale, variance, noise)
        m = m[0]
        v = v[0]
        sigma = np.sqrt(v)
        
        if sigma < 1e-8:
            return 0.0
        
        z = (f_min - m) / sigma
        ei = (f_min - m) * norm.cdf(z) + sigma * norm.pdf(z)
        return -ei  # We want to minimize negative EI to maximize EI

    def __call__(self, func):
        self.start_time = time.time()
        dim = self.dim
        budget = self.budget
        
        # Hyperparameters for GP
        # Domain is [-5, 5]^5. Distance scale ~ 1.0 seems reasonable.
        length_scale = 1.0
        variance = 1.0
        noise = 1e-6
        
        # Initial points: Halton sequence for good coverage
        n_init = 10
        X_init = self._halton_sequence(n_init, dim)
        # Scale from [0,1] to [-5,5]
        X_init = X_init * 10.0 - 5.0
        
        X = X_init.copy()
        Y = np.array([func(x) for x in X_init])
        
        f_min = np.min(Y)
        
        calls_made = n_init
        
        while calls_made < budget:
            # Check time
            if time.time() - self.start_time > self.max_time:
                break
            
            # Find next point using EI
            # Strategy: Multi-start local optimization on EI
            # 1. Sample random points to find a good starting region
            n_samples = 50
            X_samples = np.random.uniform(-5, 5, (n_samples, dim))
            
            # Compute EI for samples to find best candidate
            # We need mu and var for all samples to avoid calling _gp_predict in a loop inefficiently
            # But _gp_predict handles batches.
            mu_s, var_s = self._gp_predict(X_samples, X, Y, length_scale, variance, noise)
            
            # Calculate EI for all samples
            sigma_s = np.sqrt(var_s)
            z_s = (f_min - mu_s) / sigma_s
            ei_s = (f_min - mu_s) * norm.cdf(z_s) + sigma_s * norm.pdf(z_s)
            
            # Find the sample with highest EI
            best_idx = np.argmax(ei_s)
            x_start = X_samples[best_idx]
            
            # Refine with local optimization
            # Objective: minimize -EI
            def objective(x):
                return self._ei(x, f_min, None, None, length_scale, variance, noise, X, Y)
            
            # L-BFGS-B optimization
            res = minimize(objective, x_start, method='L-BFGS-B', bounds=[(-5, 5)] * dim, 
                           options={'maxiter': 20, 'ftol': 1e-6})
            
            x_next = res.x
            # Clip to ensure bounds
            x_next = np.clip(x_next, -5, 5)
            
            # Evaluate
            y_next = func(x_next)
            calls_made += 1
            
            # Update
            X = np.vstack([X, x_next])
            Y = np.append(Y, y_next)
            
            if y_next < f_min:
                f_min = y_next

        # Find best point
        best_idx = np.argmin(Y)
        return X[best_idx], Y[best_idx]
