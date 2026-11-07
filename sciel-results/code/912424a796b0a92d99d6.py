"""Bayesian Optimization with Gaussian Process surrogate for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


class Optimizer:
    """Bayesian Optimization using a Gaussian Process surrogate and UCB acquisition."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        lb = np.full(dim, -5.0)
        ub = np.full(dim, 5.0)

        # Initialize with a small set of diverse points
        rng = np.random.default_rng(42)
        
        # Start with origin and a few random points
        X = np.zeros((0, dim))
        Y = np.zeros(0)
        
        def add_point(x, f):
            nonlocal X, Y
            X = np.vstack([X, x.reshape(1, -1)])
            Y = np.append(Y, f)
        
        # Initial points: center, corners of a smaller box, and some random
        init_points = [
            np.zeros(dim),
            np.ones(dim) * 2.0,
            np.ones(dim) * -2.0,
            np.ones(dim) * 3.0,
            np.ones(dim) * -3.0,
        ]
        for p in init_points:
            p = np.clip(p, lb, ub)
            fp = func(p)
            add_point(p, fp)
        
        # Add a few random points
        for _ in range(3):
            p = rng.uniform(lb, ub)
            fp = func(p)
            add_point(p, fp)

        calls_made = len(X)
        
        # GP Hyperparameters
        length_scale = 1.0
        signal_var = 1.0
        noise_var = 1e-4
        
        # Normalize Y for numerical stability
        # We'll work with raw Y but standardize inside GP computations
        
        def gp_predict(X_new, X_train, Y_train, length_scale, signal_var, noise_var):
            """
            Compute mean and variance of GP at X_new given training data.
            X_train: (n, dim), Y_train: (n,)
            X_new: (m, dim)
            """
            n = len(X_train)
            m = len(X_new)
            d = X_train.shape[1]
            
            # Standardize X for kernel computation to avoid numerical issues
            # Actually, let's just use the raw values but be careful with scales
            # The domain is [-5, 5], so scales are reasonable.
            
            # RBF Kernel
            def rbf(K1, K2, ls):
                # K1: (n, d), K2: (m, d)
                # Compute squared distances
                # Use broadcasting
                # dist^2 = sum(K1^2) + sum(K2^2) - 2*K1@K2.T
                sum1 = np.sum(K1 ** 2, axis=1, keepdims=True)
                sum2 = np.sum(K2 ** 2, axis=1, keepdims=True).T
                dist2 = sum1 + sum2 - 2 * K1 @ K2.T
                # Clip to avoid negative due to floating point
                dist2 = np.maximum(dist2, 0)
                return np.exp(-0.5 * dist2 / (ls ** 2))
            
            K = rbf(X_train, X_train, length_scale)
            K += noise_var * np.eye(n)
            
            # Cholesky decomposition for stability
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # Add jitter
                K += 1e-6 * np.eye(n)
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    # Fallback to eigendecomposition or just use pseudo-inverse
                    # For now, return prior
                    return np.zeros(m), np.ones(m) * signal_var
            
            # Solve for alpha = K^-1 Y
            # L L^T Y = K Y => L^T v = K Y => L v = K Y
            K_y = K @ Y_train
            v = np.linalg.solve(L, K_y)
            alpha = np.linalg.solve(L.T, v)
            
            # Predict
            K_s = rbf(X_new, X_train, length_scale)
            mu = K_s @ alpha
            v_s = np.linalg.solve(L, K_s.T)
            var = signal_var - np.sum(v_s ** 2, axis=0)
            var = np.maximum(var, 1e-12)
            
            return mu, var

        # Acquisition Function: UCB (Upper Confidence Bound)
        # We want to MINIMIZE, so we look for low values.
        # Standard UCB for minimization: mu - beta * sigma
        # We want to maximize this acquisition function to find the next point.
        
        def acquisition(X_new, X_train, Y_train, beta):
            mu, var = gp_predict(X_new, X_train, Y_train, length_scale, signal_var, noise_var)
            sigma = np.sqrt(var)
            # Minimization: we prefer low mu and high uncertainty
            # Score = -mu + beta * sigma (we maximize this)
            return -mu + beta * sigma

        # Adaptive beta
        # Start with higher exploration, decay over time
        total_budget = budget
        current_call = calls_made
        
        # Main loop
        while calls_made < budget:
            # Determine beta based on progress
            # Linear decay from 2.0 to 0.5
            progress = (calls_made - len(X) + 1) / (budget - len(X) + 1) if budget > len(X) else 0
            # Actually, let's use the fraction of budget used
            frac_used = calls_made / budget
            beta = 2.0 - 1.5 * frac_used
            
            # Generate candidate points
            # 1. Random points
            n_random = 200
            X_rand = rng.uniform(lb, ub, (n_random, dim))
            
            # 2. Points around current best
            idx_best = np.argmin(Y)
            x_best = X[idx_best]
            n_local = 100
            # Local search around best with decreasing radius
            radius = 1.0 * (1.0 - frac_used) + 0.1
            X_local = x_best + radius * rng.standard_normal((n_local, dim))
            X_local = np.clip(X_local, lb, ub)
            
            # 3. Points around other good solutions (top 3)
            X_good = []
            top_k = min(3, len(X))
            top_indices = np.argsort(Y)[:top_k]
            for idx in top_indices:
                x_g = X[idx]
                n_g = 50
                X_g = x_g + 0.5 * (1.0 - frac_used) * rng.standard_normal((n_g, dim))
                X_g = np.clip(X_g, lb, ub)
                X_good.append(X_g)
            if X_good:
                X_good = np.vstack(X_good)
            else:
                X_good = np.zeros((0, dim))
            
            # Combine candidates
            X_cand = np.vstack([X_rand, X_local, X_good])
            
            # Evaluate acquisition function
            acq = acquisition(X_cand, X, Y, beta)
            
            # Select best candidate
            idx_max = np.argmax(acq)
            x_next = X_cand[idx_max]
            
            # Evaluate function
            f_next = func(x_next)
            calls_made += 1
            
            # Update training data
            X = np.vstack([X, x_next.reshape(1, -1)])
            Y = np.append(Y, f_next)
            
            # Optional: Update hyperparameters periodically
            # For simplicity and speed, we keep them fixed or adapt slowly
            # A simple adaptation: if the best value hasn't improved in a while, increase exploration
            # But with fixed budget 500, let's keep it simple.
            
            # Early stopping if we think we're close?
            # Hard to know f_opt. But if variance is very low and mu is stable, maybe stop?
            # The task says "repeating the last n_i if fewer calls are made", so making all 500 is good.
            # However, if we hit the same point or very close, we might waste calls.
            # The GP will assign low variance to known points, so UCB will avoid them unless beta is high.
            
        return X, Y
# EVOLVE-BLOCK-END
