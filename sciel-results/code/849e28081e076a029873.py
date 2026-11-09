"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math

class Optimizer:
    """Surrogate-Assisted Optimizer using Gaussian Processes."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin
        
        dim = self.dim
        total_budget = self.budget
        
        # State
        X = []
        F = []
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # GP Parameters
        # We use a Matern 5/2 kernel which is C2 continuous, good for smooth-ish functions
        # Büche-Rastrigin is not perfectly smooth but has structure.
        # We will fit hyperparameters in the first phase or use fixed reasonable ones.
        # Given the small budget (500), fitting hyperparameters rigorously is risky/slow.
        # We'll use a fixed kernel with length scales initialized to cover the domain.
        # Domain is [-5, 5]^5.
        
        # Kernel: Matern 5/2
        # k(x, x') = sigma_f^2 * (1 + sqrt(5)*r + 5/3*r^2) * exp(-sqrt(5)*r)
        # r = ||x - x'|| / l
        # We'll use an ARD (Automatic Relevance Determination) kernel with per-dimension length scales.
        # Initial length scales: 1.0 (covers ~20% of range, allows local detail)
        # Initial signal variance: 1.0 (normalized later or scaled)
        
        # Actually, for robustness and speed, let's use a simpler approach:
        # 1. Initial random sampling (LHS) to get a spread of points.
        # 2. Fit a simple GP with fixed hyperparameters or quick optimization.
        # 3. Sequential EI.
        
        # To avoid complex hyperparameter optimization which might eat time/budget,
        # we will use a standard GP with:
        # - Matern 5/2 kernel
        # - ARD length scales initialized to 1.0
        # - Signal variance initialized to 1.0
        # - Noise variance small (1e-6)
        
        # However, Büche-Rastrigin has a specific structure. 
        # A better strategy: 
        # Phase 1: 100 calls. Use Latin Hypercube Sampling (LHS) for diversity.
        # Phase 2: 400 calls. Use GP + EI.
        
        # We need to handle the GP prediction efficiently.
        # n=100 points in 5D is small enough for exact GP.
        
        def matern52_kernel(X1, X2, lengths, sigma_f):
            # X1: (n1, dim), X2: (n2, dim)
            # lengths: (dim,)
            # Compute pairwise distances
            # d^2 = sum_j (x1_j - x2_j)^2 / l_j^2
            X1_scaled = X1 / lengths
            X2_scaled = X2 / lengths
            # Use broadcasting
            # d2[i, j] = sum_k (X1_scaled[i, k] - X2_scaled[j, k])^2
            d2 = np.sum(X1_scaled**2, axis=1)[:, None] + np.sum(X2_scaled**2, axis=1)[None, :] - 2 * X1_scaled @ X2_scaled.T
            d2 = np.maximum(d2, 0)
            d = np.sqrt(d2)
            
            r = d
            kernel = sigma_f**2 * (1 + np.sqrt(5)*r + 5/3*r**2) * np.exp(-np.sqrt(5)*r)
            return kernel

        def predict_gp(X_new, lengths, sigma_f, noise_var):
            # X: (n, dim), F: (n,)
            n = len(X)
            if n == 0:
                return np.zeros(len(X_new)), np.zeros(len(X_new))
            
            K = matern52_kernel(np.array(X), np.array(X), lengths, sigma_f) + noise_var * np.eye(n)
            Ks = matern52_kernel(np.array(X), np.array(X_new), lengths, sigma_f)
            
            # Solve K alpha = F
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, np.array(F)))
            except:
                # Fallback if Cholesky fails
                alpha = np.linalg.solve(K + 1e-8 * np.eye(n), np.array(F))
                L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
            
            # Mean: Ks.T @ alpha
            mu = Ks.T @ alpha
            
            # Variance: k(x_new, x_new) - Ks.T @ K^-1 @ Ks
            # v = K^-1 @ Ks
            try:
                v = np.linalg.solve(L.T, np.linalg.solve(L, Ks))
            except:
                v = np.linalg.solve(K + 1e-8 * np.eye(n), Ks)
            
            var = sigma_f**2 - np.sum(v * Ks, axis=0)
            var = np.maximum(var, 1e-10)
            
            return mu, var

        # Phase 1: Initial Exploration
        # Use LHS for better coverage than pure random
        n_init = 100
        # Simple LHS implementation
        def lhs(n, dim):
            # Returns points in [0, 1]^dim
            pts = []
            for i in range(n):
                # For each dimension, take the i-th point from a permuted grid
                # This is a simplified LHS
                pass
            # Simpler: just use random for speed, but ensure spread?
            # Actually, pure random is often fine for 100 points in 5D.
            # Let's use random to keep it simple and fast.
            return np.random.uniform(-5, 5, (n, dim))
        
        # Initialize with random points
        rng = np.random.default_rng(42)
        X_init = rng.uniform(-5, 5, (n_init, dim))
        
        for i in range(n_init):
            if calls_made >= total_budget:
                break
            if time.time() >= deadline:
                break
            
            x = X_init[i]
            f = func(x)
            calls_made += 1
            X.append(x)
            F.append(f)
            
            if f < best_f:
                best_f = f
                best_x = x.copy()
        
        # Phase 2: Surrogate-Guided Optimization
        # We will use EI acquisition.
        # To make it efficient, we will evaluate 1 point at a time (sequential).
        # But calculating EI for a grid of candidates is expensive if the grid is large.
        # Strategy:
        # 1. Maintain a set of candidate points.
        # 2. Or, sample candidates from the GP posterior? No, EI needs the mean/var.
        # 3. Standard approach: Sample a grid of points (e.g., 1000 points) in the domain,
        #    predict mean/var for all, calculate EI, pick best, evaluate, update GP.
        
        # To avoid the cost of re-solving the GP system for a large grid repeatedly,
        # we can limit the number of candidate points or use a coarser grid.
        # 5D domain. A grid of 20^5 = 3.2M points is too big.
        # Let's use random sampling of candidates.
        # At each step:
        # 1. Generate M random candidate points in [-5, 5]^5.
        # 2. Predict GP mean/var for these M points.
        # 3. Calculate EI.
        # 4. Pick the point with max EI.
        # 5. Evaluate it.
        # 6. Add to X, F.
        # 7. Repeat.
        
        # M = 500 candidates per step.
        # 400 steps * 500 candidates * GP prediction cost.
        # GP prediction for 500 points given 100-500 training points:
        # K is (n_train, n_train). n_train grows up to 500.
        # Ks is (n_train, 500).
        # Solving (500x500) system 400 times might be slow in Python/NumPy if not optimized.
        # 500^3 ~ 125M ops per solve. 400 solves ~ 50B ops. Might be too slow for 280s.
        
        # Optimization: 
        # - Use a smaller candidate pool M=100.
        # - Or, update GP less frequently? No, we need fresh info.
        # - Or, use a simpler kernel?
        # - Or, batch the candidate evaluation?
        
        # Let's try M=100 candidates per step.
        # Also, we can stop early if EI is very low everywhere (convergence).
        
        # Hyperparameters:
        # We'll keep lengths fixed at 1.0 and sigma_f fixed at 1.0 for now.
        # If the function values are large, sigma_f=1 might be too small.
        # Let's normalize F or estimate sigma_f from initial data.
        
        # Estimate sigma_f from initial 100 points
        F_arr = np.array(F)
        sigma_f_init = np.std(F_arr) + 1e-6
        # If std is 0, use 1.0
        if sigma_f_init < 1e-3:
            sigma_f_init = 1.0
            
        # Initial lengths: 1.0
        lengths = np.ones(dim)
        noise_var = 1e-6 * sigma_f_init**2
        
        # EI Calculation
        def expected_improvement(mu, var, f_best, xi=0.0):
            # f_best is the current best (minimum)
            # EI = (f_best - mu - xi) * Phi((f_best - mu - xi) / sigma) + sigma * phi((f_best - mu - xi) / sigma)
            # Note: For minimization, we want f < f_best.
            # Standard EI for minimization:
            # d = (f_best - mu) / sigma
            # EI = sigma * (d * Phi(d) + phi(d))
            # Where Phi is CDF of standard normal, phi is PDF.
            # If sigma is small, use limit.
            
            sigma = np.sqrt(var)
            # Avoid division by zero
            sigma = np.maximum(sigma, 1e-8)
            
            d = (f_best - mu) / sigma
            # Phi(d) = 0.5 * (1 + erf(d / sqrt(2)))
            phi_d = np.exp(-0.5 * d**2) / np.sqrt(2 * np.pi)
            Phi_d = 0.5 * (1 + np.erf(d / np.sqrt(2)))
            
            ei = sigma * (d * Phi_d + phi_d)
            
            # Clip to 0 if negative due to numerical issues
            ei = np.maximum(ei, 0)
            return ei

        # Loop for remaining budget
        M_candidates = 100
        
        while calls_made < total_budget:
            if time.time() >= deadline:
                break
            
            # Generate candidates
            # To encourage exploration of promising areas, we can bias sampling towards current best?
            # Or just uniform? Uniform is safer for global search.
            # Let's mix: 50% uniform, 50% local around best.
            X_cand = []
            n_uniform = M_candidates // 2
            n_local = M_candidates - n_uniform
            
            # Uniform
            X_cand.append(rng.uniform(-5, 5, (n_uniform, dim)))
            
            # Local around best
            if best_x is not None:
                # Scale with current best uncertainty? Or fixed?
                # Let's use a fixed radius that shrinks?
                # For now, radius 1.0
                X_local = best_x + rng.normal(0, 1.0, (n_local, dim))
                X_local = np.clip(X_local, -5, 5)
                X_cand.append(X_local)
            
            X_cand = np.vstack(X_cand)
            
            # Predict
            mu, var = predict_gp(X_cand, lengths, sigma_f_init, noise_var)
            
            # EI
            ei = expected_improvement(mu, var, best_f)
            
            # Pick best candidate
            idx = np.argmax(ei)
            x_new = X_cand[idx]
            
            # Evaluate
            f_new = func(x_new)
            calls_made += 1
            
            X.append(x_new)
            F.append(f_new)
            
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()
                
                # Optional: Update sigma_f or lengths?
                # Keeping them fixed is safer for stability.
            
            # If EI is very low, we might be stuck.
            # If max_ei < threshold, we could break or restart?
            # For now, just continue.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
