import numpy as np
from scipy.stats import norm
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.X = []
        self.Y = []
        
        # GP Hyperparameters
        # Initial guesses
        self.length_scale = 1.5
        self.noise = 1e-2
        self.signal_var = 1.0

    def __call__(self, func):
        # 1. Initialization
        # Use Halton sequence for deterministic space-filling
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

        n_init = min(20, self.budget)
        X_init = halton(self.dim, n_init)
        X_init = X_init * 10 - 5
        
        self.X = []
        self.Y = []
        for x in X_init:
            y = func(x)
            self.X.append(x)
            self.Y.append(y)
            
        self.X = np.array(self.X)
        self.Y = np.array(self.Y)
        
        # Normalize Y for GP stability
        self.Y_mean = np.mean(self.Y)
        self.Y_std = np.std(self.Y)
        if self.Y_std < 1e-8:
            self.Y_std = 1.0
        self.Y_norm = (self.Y - self.Y_mean) / self.Y_std

        # Pre-allocate for speed? Not strictly necessary, but good practice.
        
        # Helper: Matern 5/2 Kernel
        def matern52_kernel(X1, X2, l):
            # X1: (N, D), X2: (M, D)
            sq_dists = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1)[np.newaxis, :] - 2 * X1 @ X2.T
            np.maximum(sq_dists, 0, out=sq_dists)
            d = np.sqrt(sq_dists) / l
            return self.signal_var**2 * (1 + np.sqrt(5)*d + 5/3*d**2) * np.exp(-np.sqrt(5)*d)

        # Helper: GP Prediction
        def gp_predict(X_test):
            # Uses self.X, self.Y_norm, self.length_scale, self.noise
            N = len(self.X)
            K = matern52_kernel(self.X, self.X, self.length_scale) + self.noise * np.eye(N)
            # Cholesky decomposition for stability and speed
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # Fallback to inv if Cholesky fails
                K_inv = np.linalg.inv(K)
                K_s = matern52_kernel(X_test, self.X, self.length_scale)
                mean = K_s @ K_inv @ self.Y_norm
                v = K_inv @ K_s.T
                var = self.signal_var**2 - np.sum(K_s * v, axis=1)
                var = np.maximum(var, 1e-8)
                return mean, np.sqrt(var)
            
            # Solve L v = Y_norm
            v = np.linalg.solve(L, self.Y_norm)
            K_s = matern52_kernel(X_test, self.X, self.length_scale)
            u = np.linalg.solve(L, K_s.T) # (N, M)
            mean = u.T @ v # (M,)
            var = self.signal_var**2 - np.sum(u**2, axis=0)
            var = np.maximum(var, 1e-8)
            return mean, np.sqrt(var)

        # Helper: Expected Improvement
        def expected_improvement(mean, std, y_best):
            z = (y_best - mean) / std
            return (y_best - mean) * norm.cdf(z) + std * norm.pdf(z)

        # Helper: Hyperparameter Optimization (Quick)
        def optimize_hyperparams():
            # Optimize log(length_scale) and log(noise)
            # Using a small grid or a few NM steps
            # For speed, let's use a small grid search over log-l and log-noise
            # l in [0.5, 3.0], noise in [1e-3, 1e-1]
            best_ll = -np.inf
            best_l = self.length_scale
            best_n = self.noise
            
            # Coarse grid
            l_vals = np.logspace(np.log10(0.5), np.log10(3.0), 5)
            n_vals = np.logspace(np.log10(1e-3), np.log10(1e-1), 5)
            
            for l in l_vals:
                for n in n_vals:
                    try:
                        K = matern52_kernel(self.X, self.X, l) + n * np.eye(len(self.X))
                        L = np.linalg.cholesky(K)
                        # Log likelihood: -0.5 * (log det K + y^T K^-1 y)
                        log_det = 2 * np.sum(np.log(np.diag(L)))
                        alpha = np.linalg.solve(L, self.Y_norm)
                        alpha = np.linalg.solve(L.T, alpha)
                        ll = -0.5 * (log_det + self.Y_norm @ alpha)
                        if ll > best_ll:
                            best_ll = ll
                            best_l = l
                            best_n = n
                    except:
                        continue
            
            self.length_scale = best_l
            self.noise = best_n

        # Main Loop
        n_calls = n_init
        while n_calls < self.budget:
            # Optimize hyperparameters every 10 steps
            if (n_calls - n_init) % 10 == 0 and n_calls > n_init:
                optimize_hyperparams()

            y_best_norm = np.min(self.Y_norm)
            
            # Multi-fidelity Acquisition Maximization
            # Level 1: Coarse Grid
            # 10 points per dimension -> 10^5 = 100,000 points
            # This might be too slow if GP is not optimized, but let's try 8^5 = 32,768
            n_grid = 8
            grid_points = np.meshgrid(*[np.linspace(-5, 5, n_grid) for _ in range(self.dim)])
            grid_points = np.array(grid_points).T.reshape(-1, self.dim)
            
            means, stds = gp_predict(grid_points)
            eis = expected_improvement(means, stds, y_best_norm)
            
            # Find top K candidates
            k_top = 5
            top_indices = np.argsort(eis)[-k_top:]
            top_points = grid_points[top_indices]
            
            # Level 2: Fine Grid around top candidates
            # For each top point, create a small local grid
            # Box size: 0.5 units around the point
            # 5 points per dimension -> 5^5 = 3125 points per center
            # Total candidates: 5 * 3125 = 15,625
            local_candidates = []
            for p in top_points:
                # Define local bounds
                lower = np.maximum(-5, p - 0.5)
                upper = np.minimum(5, p + 0.5)
                local_grid = np.meshgrid(*[np.linspace(lower[i], upper[i], 5) for i in range(self.dim)])
                local_grid = np.array(local_grid).T.reshape(-1, self.dim)
                local_candidates.append(local_grid)
            
            local_candidates = np.vstack(local_candidates)
            
            # Predict on local candidates
            means_loc, stds_loc = gp_predict(local_candidates)
            eis_loc = expected_improvement(means_loc, stds_loc, y_best_norm)
            
            # Select best
            best_idx_loc = np.argmax(eis_loc)
            x_new = local_candidates[best_idx_loc]
            
            # Evaluate
            y_new = func(x_new)
            
            # Update state
            self.X = np.vstack([self.X, x_new])
            self.Y = np.vstack([self.Y, y_new])
            
            # Update normalized Y
            self.Y_mean = np.mean(self.Y)
            self.Y_std = np.std(self.Y)
            if self.Y_std < 1e-8:
                self.Y_std = 1.0
            self.Y_norm = (self.Y - self.Y_mean) / self.Y_std
            
            n_calls += 1

        best_idx = np.argmin(self.Y)
        return self.X[best_idx], self.Y[best_idx]
