"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Bayesian Optimizer using Gaussian Process surrogate and Expected Improvement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s limit, aim to finish by 300s
        deadline = start_time + 300.0
        
        # State tracking
        best_x = None
        best_f = float('inf')
        calls_used = 0
        
        # Data storage
        X = []
        Y = []

        # Wrapped function to track calls and enforce budget
        def wrapped_func(x):
            nonlocal calls_used, best_x, best_f
            if calls_used >= self.budget:
                return 1e10 # Return high value to stop optimizer if budget hit
            
            # Ensure x is a valid numpy array of correct shape
            x_arr = np.asarray(x, dtype=float)
            x_clipped = np.clip(x_arr, -5, 5)
            
            try:
                fx = float(func(x_clipped))
            except Exception:
                fx = 1e10
            
            calls_used += 1
            if fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
                
            X.append(x_clipped.copy())
            Y.append(fx)
            
            return fx

        # 1. Initial Sampling (Space Filling)
        # Use a simple Latin Hypercube or random points to initialize the GP
        rng = np.random.default_rng(42)
        n_init = 15
        # Ensure we don't exceed budget
        n_init = min(n_init, self.budget)
        
        # Generate initial points
        # Simple random is okay for 5D, but LHS is better. Let's do a mix.
        # Origin is a common trap/start for Rosenbrock, but shift is unknown.
        # Let's use random points in [-5, 5]^dim
        X_init = rng.uniform(-5, 5, size=(n_init, self.dim))
        
        # Evaluate initial points
        for x0 in X_init:
            if time.time() > deadline:
                break
            if calls_used >= self.budget:
                break
            wrapped_func(x0)

        if best_x is None:
             # Fallback if something went wrong
             best_x = np.zeros(self.dim)
             best_f = func(best_x)
             calls_used += 1
             X.append(best_x.copy())
             Y.append(best_f)

        # 2. Bayesian Optimization Loop
        # We implement a simple GP with RBF kernel and EI acquisition
        # to avoid dependency on heavy external libraries if not strictly necessary, 
        # but SciPy is available. However, implementing a robust GP from scratch 
        # in 5D with 500 calls requires care.
        
        # Let's use a simplified GP implementation with RBF kernel.
        # K(x, x') = sigma_f^2 * exp(-0.5 * ||x - x'||^2 / l^2)
        
        sigma_f = 1.0
        l = 1.0
        noise = 1e-8
        
        def compute_kernel(X1, X2, sigma_f, l, noise):
            # X1: (n1, d), X2: (n2, d)
            # Returns (n1, n2)
            diff = X1[:, np.newaxis, :] - X2[np.newaxis, :, :]
            sq_dist = np.sum(diff**2, axis=2)
            K = sigma_f**2 * np.exp(-0.5 * sq_dist / (l**2))
            if X1.shape[0] == X2.shape[0] and X1 is X2: # Add noise only for training data
                K += noise * np.eye(X1.shape[0])
            return K

        def predict_gp(X_new, X_train, Y_train, sigma_f, l, noise):
            K = compute_kernel(X_train, X_train, sigma_f, l, noise)
            K_star = compute_kernel(X_new, X_train, sigma_f, l, 0) # No noise for test
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            K_star_T = K_star.T
            v = np.linalg.solve(L, K_star_T)
            mu = v.T @ alpha
            var = sigma_f**2 - np.sum(v**2, axis=0)
            var = np.maximum(var, 1e-10) # Numerical stability
            return mu, var

        def expected_improvement(mu, var, best_f, xi=0.01):
            # mu, var: arrays of shape (n_candidates,)
            # best_f: scalar
            # xi: exploration/exploitation trade-off
            if var <= 0:
                return np.zeros_like(mu)
            
            es = (best_f - xi - mu) / np.sqrt(var)
            # Standard normal CDF and PDF
            # Using scipy.stats.norm
            from scipy.stats import norm
            z = es * norm.cdf(es) + norm.pdf(es)
            return z * np.sqrt(var)

        # Main Loop
        # We will generate a batch of candidates, evaluate EI, pick the best, and update.
        # To speed up, we can pick multiple points at once (batch BO) or sequentially.
        # Sequential is more robust for GP updates.
        
        while calls_used < self.budget and time.time() < deadline:
            # Generate candidates
            # Use a mix of random and points near the current best
            n_cands = 200
            
            # Random candidates
            X_rand = rng.uniform(-5, 5, size=(n_cands // 2, self.dim))
            
            # Local search candidates around best_x
            # If best_x is valid
            if best_x is not None:
                # Sample around best_x with decreasing radius?
                # For now, just a Gaussian cloud
                X_local = best_x + rng.normal(0, 1.0, size=(n_cands // 2, self.dim))
                X_local = np.clip(X_local, -5, 5)
                X_cands = np.vstack([X_rand, X_local])
            else:
                X_cands = X_rand

            # Predict GP
            X_train = np.array(X)
            Y_train = np.array(Y)
            
            try:
                mu, var = predict_gp(X_cands, X_train, Y_train, sigma_f, l, noise)
                ei = expected_improvement(mu, var, best_f)
                
                # Select best candidate
                idx = np.argmax(ei)
                x_next = X_cands[idx]
                
            except Exception:
                # Fallback to random if GP fails
                x_next = rng.uniform(-5, 5, self.dim)

            # Evaluate
            wrapped_func(x_next)
            
            # Adaptive Hyperparameters (Simple)
            # If we are stagnating, increase exploration (increase length scale or noise)
            # This is a heuristic. For now, keep fixed.
            
            # Check time
            if time.time() > deadline:
                break

        return best_x, best_f

# EVOLVE-BLOCK-END
