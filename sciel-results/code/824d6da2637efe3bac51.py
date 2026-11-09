import numpy as np
import time
from scipy.optimize import minimize
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.deadline = time.time() + 320.0  # Safety margin from 330s limit

    def __call__(self, func):
        # Initialize data structures
        self.X = []
        self.Y = []
        
        # Initial sampling: Use a space-filling or random start
        # Let's start with a few random points and the origin
        rng = np.random.default_rng(42)
        
        # Add origin and some random points
        init_points = [np.zeros(self.dim)]
        for _ in range(4):
            init_points.append(rng.uniform(-5, 5, self.dim))
            
        for x in init_points:
            fx = func(x)
            self.X.append(x)
            self.Y.append(fx)
            
        best_x = self.X[0]
        best_y = self.Y[0]
        
        # GP Hyperparameters
        # Matern 5/2 kernel is good for non-smooth functions
        # length_scale: typical distance between features
        # variance: typical magnitude of f
        # We assume f is in [-100, 100] roughly, so variance ~ 1000 might be too high, let's start with 1.0 and normalize Y
        # Actually, let's normalize Y to zero mean, unit variance for the GP to be stable.
        
        def matern52_kernel(K, l):
            # K is the distance matrix
            # Matern 5/2: (1 + sqrt(5)*K + 5/3*K^2) * exp(-sqrt(5)*K)
            sqrt5 = np.sqrt(5.0)
            return (1.0 + sqrt5 * K + (5.0/3.0) * K**2) * np.exp(-sqrt5 * K)

        def compute_gp_predict(X_train, Y_train, X_test, length_scale=1.5, variance=1.0, noise=1e-10):
            """
            Computes mean and variance of GP at X_test given X_train, Y_train.
            Assumes Y_train is normalized (mean 0, var 1).
            """
            n = len(X_train)
            m = len(X_test)
            
            # Compute kernel matrices
            # K_train_train
            diff_tt = X_train[:, np.newaxis, :] - X_train[np.newaxis, :, :]
            dist_tt = np.sqrt(np.sum(diff_tt**2, axis=-1))
            K_tt = matern52_kernel(dist_tt, length_scale) * variance
            K_tt += noise * np.eye(n)
            
            # K_test_train
            diff_xt = X_test[:, np.newaxis, :] - X_train[np.newaxis, :, :]
            dist_xt = np.sqrt(np.sum(diff_xt**2, axis=-1))
            K_xt = matern52_kernel(dist_xt, length_scale) * variance
            
            # K_test_test (for variance)
            diff_xx = X_test[:, np.newaxis, :] - X_test[np.newaxis, :, :]
            dist_xx = np.sqrt(np.sum(diff_xx**2, axis=-1))
            K_xx = matern52_kernel(dist_xx, length_scale) * variance
            
            # Solve for alpha
            try:
                L = np.linalg.cholesky(K_tt)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
                
                # Mean
                mu = K_xt.T @ alpha
                
                # Variance
                v = np.linalg.solve(L, K_xt.T)
                var = np.diag(K_xx) - np.sum(v**2, axis=0)
                var = np.maximum(var, 0.0) # Numerical stability
            except np.linalg.LinAlgError:
                # Fallback if Cholesky fails (shouldn't happen with noise)
                mu = np.zeros(m)
                var = np.ones(m)
                
            return mu, var

        # Main Loop
        for i in range(self.budget - len(self.X)):
            if time.time() > self.deadline:
                break
                
            # Normalize Y for GP
            y_mean = np.mean(self.Y)
            y_std = np.std(self.Y)
            if y_std < 1e-10:
                y_std = 1.0
            Y_norm = (np.array(self.Y) - y_mean) / y_std
            
            X_train = np.array(self.X)
            Y_train = Y_norm
            
            # Generate candidates
            # Mix of:
            # 1. Local perturbations around best_x
            # 2. Random points in the box
            # 3. Points suggested by the GP (max uncertainty)
            
            candidates = []
            
            # 1. Local search around best
            for _ in range(20):
                # Gaussian perturbation
                scale = 1.0
                x_cand = best_x + rng.normal(0, scale, self.dim)
                x_cand = np.clip(x_cand, -5, 5)
                candidates.append(x_cand)
                
            # 2. Random exploration
            for _ in range(20):
                x_cand = rng.uniform(-5, 5, self.dim)
                candidates.append(x_cand)
                
            # 3. GP-guided exploration (Max Variance)
            # Sample a grid or random points and pick high variance ones
            # To save time, just pick a few random and let the acquisition handle it,
            # or use a coarse grid. Let's stick to random candidates evaluated by EI/UCB.
            
            candidates = np.array(candidates)
            
            # Evaluate Acquisition Function (UCB)
            # UCB = mu + beta * sigma
            # We want to MINIMIZE f.
            # So we want LOW mu and HIGH sigma.
            # Acquisition = -mu + beta * sigma
            
            mu, sigma = compute_gp_predict(X_train, Y_train, candidates, length_scale=1.5, variance=1.0)
            sigma = np.sqrt(sigma)
            
            # Adaptive beta: start high, decrease as we find better points?
            # Or just fixed. Let's use a fixed beta that balances exploration.
            # If best_y is far from 0 (normalized), we might need more exploration.
            # Let's use beta = 2.0 for now.
            beta = 2.0
            acq = -mu + beta * sigma
            
            # Select best candidate based on acquisition
            idx = np.argmax(acq)
            x_next = candidates[idx]
            
            # Evaluate
            f_next = func(x_next)
            
            # Update data
            self.X.append(x_next)
            self.Y.append(f_next)
            
            if f_next < best_y:
                best_y = f_next
                best_x = x_next.copy()
                
        return best_x, best_y
