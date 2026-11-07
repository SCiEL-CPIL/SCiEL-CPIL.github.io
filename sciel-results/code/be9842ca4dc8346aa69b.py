import numpy as np
import time
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325.0  # 5s safety margin before 330s kill
        
        # 1. Initial Exploration Phase
        # For Bayesian Optimization, we need an initial set of points to fit the surrogate.
        # In 5D, 10-20 points is a reasonable start.
        n_init = min(15, self.budget)
        
        # Use a quasi-random sequence (Sobol-like via Halton or just random with jitter)
        # For simplicity and robustness, we use a Latin Hypercube Sample (LHS) 
        # or just random points spread out.
        # Given the box [-5, 5], let's use a grid-like initial spread to cover the space.
        # Actually, random is fine for GP initialization if we use a decent kernel.
        # Let's use a mix: center + random corners/edges to ensure global coverage.
        
        rng = np.random.default_rng()
        
        X_train = []
        Y_train = []
        
        # Add origin
        x0 = np.zeros(self.dim)
        f0 = func(x0)
        X_train.append(x0)
        Y_train.append(f0)
        self.calls += 1
        
        # Add random points
        for _ in range(n_init - 1):
            x = rng.uniform(-5, 5, size=self.dim)
            f = func(x)
            X_train.append(x)
            Y_train.append(f)
            self.calls += 1
            
            if self.calls >= self.budget:
                break

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        
        # Update best
        idx_best = np.argmin(Y_train)
        self.best_x = X_train[idx_best].copy()
        self.best_f = Y_train[idx_best]

        if self.calls >= self.budget:
            return self.best_x, self.best_f

        # 2. Bayesian Optimization Loop
        # We will use a simple Matern 5/2 kernel.
        # To handle the 500 budget, we update the GP after every new point (or every few points).
        # For dim=5, solving a 500x500 system is O(500^3) ~ 125M ops, which is fine in Python/NumPy if done ~100 times.
        # However, fitting a GP from scratch every step is O(N^3). 
        # With N=500, that's 125M operations. 500 steps * 125M = 62.5B ops. This might be too slow in pure Python/NumPy loops if not optimized.
        # Optimization: Use a fixed set of "inducing points" or just re-fit on all data but limit the GP complexity.
        # Alternatively, since dim is small (5), we can use a simpler surrogate like Kriging with a fixed kernel and just re-fit.
        # Let's try re-fitting on all data. If it's too slow, we will fall back to a simpler method.
        # Actually, 500 iterations of O(N^3) with N growing to 500 is roughly sum(N^2) ~ 500^3/3 ~ 40M ops? No, Cholesky is N^3/3.
        # 500^3 / 3 = 41 million flops per iteration. 500 iterations = 20 billion flops. 
        # NumPy is fast, but Python overhead for the loop might be an issue.
        # Let's limit the GP training set size to the last 100 points or use a sparse approximation.
        # A better approach for small budgets: Update GP every 5-10 steps, or use a simpler EI acquisition with a fixed kernel width.
        
        # Let's implement a robust loop that checks time.
        
        # Hyperparameters for Matern 5/2
        # l: length scale, sigma: signal variance, noise: observation noise
        # We will optimize these roughly or fix them.
        # Fixed hyperparameters are safer for avoiding optimization failures.
        l = 1.0
        sigma_f = 1.0
        noise = 1e-6
        
        def matern52_kernel(X1, X2, l):
            # X1: (N1, D), X2: (N2, D)
            # Returns (N1, N2)
            sq_dist = np.sum(X1**2, axis=1)[:, None] + np.sum(X2**2, axis=1)[None, :] - 2 * X1 @ X2.T
            sq_dist = np.maximum(sq_dist, 0)
            dist = np.sqrt(sq_dist)
            return (1 + np.sqrt(3) * dist / l) * np.exp(-np.sqrt(3) * dist / l)

        def predict_gp(X_new, X_train, Y_train, l, sigma_f, noise):
            # X_new: (M, D)
            # X_train: (N, D)
            # Y_train: (N,)
            K = matern52_kernel(X_train, X_train, l)
            K += noise * np.eye(len(X_train))
            K_s = matern52_kernel(X_new, X_train, l)
            
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # Add jitter if Cholesky fails
                K += 1e-6 * np.eye(len(X_train))
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    # Fallback to identity if still failing (should not happen)
                    return np.zeros(len(X_new)), np.ones(len(X_new))
            
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            
            mean = K_s @ alpha
            v = np.linalg.solve(L, K_s.T)
            cov_diag = sigma_f**2 - np.sum(v**2, axis=0)
            std = np.sqrt(np.maximum(cov_diag, 1e-8))
            
            return mean, std

        def expected_improvement(mean, std, best_f, xi=0.0):
            # EI for minimization
            z = (best_f - mean - xi) / (std + 1e-8)
            # Handle cases where std is very small
            ei = np.zeros_like(z)
            mask = std > 1e-8
            z_masked = z[mask]
            ei[mask] = (best_f - mean[mask] - xi) * norm_cdf(z_masked) + std[mask] * norm_pdf(z_masked)
            return ei

        def norm_cdf(x):
            return 0.5 * (1 + np.erf(x / np.sqrt(2)))

        def norm_pdf(x):
            return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)

        # Optimization Loop
        # We will search for the next point by maximizing EI over a candidate set.
        # Candidate set: Local perturbations around best points + random global points.
        
        steps_since_improve = 0
        max_steps_without_improve = 20
        
        while self.calls < self.budget:
            if time.time() > deadline - 5:
                break
                
            # Update best
            # (Already done)
            
            # Generate candidates
            # 1. Local search around current best and top few points
            # 2. Random global points to avoid getting stuck
            
            # Sort training data by f
            sorted_idx = np.argsort(Y_train)
            top_k = min(5, len(X_train))
            top_points = X_train[sorted_idx[:top_k]]
            
            candidates = []
            
            # Local perturbations
            # Use a small step size relative to the kernel length scale
            step_size = l * 0.5
            for p in top_points:
                for _ in range(5):
                    perturbation = rng.normal(0, step_size, size=self.dim)
                    x_cand = p + perturbation
                    x_cand = np.clip(x_cand, -5, 5)
                    candidates.append(x_cand)
            
            # Global random candidates
            for _ in range(10):
                x_cand = rng.uniform(-5, 5, size=self.dim)
                candidates.append(x_cand)
                
            candidates = np.array(candidates)
            
            # Evaluate GP on candidates
            mean, std = predict_gp(candidates, X_train, Y_train, l, sigma_f, noise)
            
            # Calculate EI
            ei = expected_improvement(mean, std, self.best_f)
            
            # Select best candidate
            idx_best_cand = np.argmax(ei)
            x_new = candidates[idx_best_cand]
            
            # Evaluate function
            f_new = func(x_new)
            self.calls += 1
            
            # Update training set
            X_train = np.vstack([X_train, x_new])
            Y_train = np.append(Y_train, f_new)
            
            # Update best
            if f_new < self.best_f:
                self.best_f = f_new
                self.best_x = x_new.copy()
                steps_since_improve = 0
            else:
                steps_since_improve += 1
                
            # If we are stuck for a long time, we might want to increase exploration
            # By increasing the 'xi' in EI or just relying on the random candidates.
            # The random candidates in the list above provide exploration.
            
            # Optional: Adapt length scale if we are stuck? 
            # For now, keep it simple.
            
            # Check if we should break early if time is tight
            # The loop condition handles the budget.
            
        return self.best_x, self.best_f

# Helper functions for the Optimizer class to access
# (Defined inside __call__ or as static methods if needed, but here we defined them locally)
