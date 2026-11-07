import numpy as np
from scipy.optimize import minimize
import time
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # State
        X = []
        Y = []
        
        # 1. Initial Sampling: Use Halton sequence for deterministic coverage
        def halton(dim, n, skip=0):
            def halton_index(i, base):
                f = 1.0
                r = 0.0
                while i > 0:
                    f /= base
                    r += f * (i % base)
                    i //= base
                return r
            
            points = np.zeros((n, dim))
            for d in range(dim):
                for i in range(n):
                    points[i, d] = halton_index(i + skip + 1, d + 2)
            return points

        # Use 30 initial points to better capture multimodality
        n_init = min(30, self.budget)
        X_init = halton(self.dim, n_init)
        X_init = X_init * 10 - 5
        
        for x in X_init:
            y = func(x)
            X.append(x)
            Y.append(y)
            
        X = np.array(X)
        Y = np.array(Y)
        
        # GP Parameters
        # We use Matern 5/2 kernel
        # Fixed hyperparameters for speed and stability
        length_scale = 1.2
        sigma_f = 1.0
        sigma_n = 1e-4
        
        def matern52_kernel(X1, X2, l):
            sq_dists = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1)[np.newaxis, :] - 2 * X1 @ X2.T
            np.maximum(sq_dists, 0, out=sq_dists)
            d = np.sqrt(sq_dists) / l
            return sigma_f**2 * (1 + np.sqrt(5)*d + 5/3*d**2) * np.exp(-np.sqrt(5)*d)

        def gp_predict(X_test, K_inv, Y_train, l, sf, sn):
            # K_inv is precomputed inverse of (K + sn*I)
            K_s = matern52_kernel(X_test, X_train_global, l)
            mean = K_s @ (K_inv @ Y_train)
            v = K_inv @ K_s.T
            var = sf**2 - np.sum(K_s * v, axis=1)
            var = np.maximum(var, 1e-8)
            return mean, np.sqrt(var)

        def expected_improvement(mean, std, y_best):
            z = (y_best - mean) / std
            ei = (y_best - mean) * norm.cdf(z) + std * norm.pdf(z)
            return ei

        # Main Loop
        # We will use a Latin Hypercube for acquisition maximization
        # This is much faster than Nelder-Mead for 5D
        n_lhs = 500 # Number of LHS points to sample for acquisition
        
        for i in range(self.budget - n_init):
            # Normalize Y for GP to handle different scales
            Y_mean = np.mean(Y)
            Y_std = np.std(Y)
            if Y_std < 1e-6:
                Y_std = 1.0
            Y_norm = (Y - Y_mean) / Y_std
            
            # Precompute GP components
            X_train_global = X
            K = matern52_kernel(X_train_global, X_train_global, length_scale) + sigma_n * np.eye(len(X_train_global))
            try:
                K_inv = np.linalg.inv(K)
            except np.linalg.LinAlgError:
                # Fallback if singular
                K_inv = np.linalg.pinv(K)
            
            y_best_norm = np.min(Y_norm)
            
            # Generate LHS candidates
            # Latin Hypercube Sampling
            lhs_points = np.zeros((n_lhs, self.dim))
            for d in range(self.dim):
                idx = np.arange(n_lhs)
                np.random.shuffle(idx)
                lhs_points[:, d] = (idx + 0.5) / n_lhs
            lhs_points = lhs_points * 10 - 5
            
            # Predict on LHS points
            means, stds = gp_predict(lhs_points, K_inv, Y_norm, length_scale, sigma_f, sigma_n)
            eis = expected_improvement(means, stds, y_best_norm)
            
            # Pick the point with highest EI
            best_idx = np.argmax(eis)
            x_new = lhs_points[best_idx]
            
            # Evaluate
            x_new = np.clip(x_new, -5, 5)
            y_new = func(x_new)
            
            X = np.vstack([X, x_new])
            Y = np.vstack([Y, y_new])

        return X[np.argmin(Y)], np.min(Y)
