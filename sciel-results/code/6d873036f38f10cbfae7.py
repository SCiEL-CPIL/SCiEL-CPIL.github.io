# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm


class Optimizer:
    """
    Surrogate-based optimizer (Gaussian Process) for BBOB f7.
    
    Strategy:
    1. Initialize with a Latin Hypercube sample to cover the search space.
    2. Fit a Gaussian Process surrogate to the observed data.
    3. Select the next point using Expected Improvement (EI) acquisition function.
    4. This maximizes information gain per evaluation, accelerating early convergence
       and efficiently navigating the step-like landscape of f7.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self._call_count = 0
        self._best_x = None
        self._best_f = np.inf

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin before 330s kill
        dim = self.dim
        budget = self.budget

        self._call_count = 0
        self._best_f = np.inf
        self._best_x = None

        def wrapped(x):
            self._call_count += 1
            fx = float(func(x))
            if fx < self._best_f:
                self._best_f = fx
                self._best_x = np.array(x, dtype=float)
            return fx

        rng = np.random.default_rng(42)

        # Data storage
        X = []
        Y = []

        # Helper to fit GP and get acquisition
        def fit_gp_and_acq(X_data, Y_data):
            X_data = np.array(X_data)
            Y_data = np.array(Y_data)
            
            # Normalize Y for better GP conditioning
            y_mean = Y_data.mean()
            y_std = Y_data.std()
            if y_std < 1e-8:
                y_std = 1.0
            Y_norm = (Y_data - y_mean) / y_std

            # Simple Matern kernel implementation to avoid dependency on specific GP libraries
            # K(x, x') = sigma^2 * (1 + sqrt(5)*r/l) * exp(-sqrt(5)*r/l)
            # We use a simple isotropic kernel for simplicity and speed, or RBF.
            # RBF is often faster to evaluate. Let's use RBF with length-scale optimization via a few random restarts or fixed.
            
            # For speed and robustness in dim=5, we can use a fixed kernel or simple optimization.
            # Given the tight time limit, we'll use a simple RBF kernel with fixed hyperparameters 
            # or a very quick fit.
            
            # Let's implement a lightweight GP fit using RBF kernel.
            # K(x, x') = exp(-0.5 * ||x-x'||^2 / l^2)
            
            # To keep it fast, we'll fix the length scale or use a small grid search.
            # Let's try a few length scales.
            
            best_score = -np.inf
            best_mu = None
            best_cov = None
            
            # Precompute pairwise distances
            X_sq = np.sum(X_data ** 2, axis=1)
            D = X_sq[:, None] + X_sq[None, :] - 2 * X_data @ X_data.T
            D = np.maximum(D, 0)
            
            # Try a few length scales
            for l in [0.5, 1.0, 2.0]:
                K = np.exp(-D / (2 * l ** 2))
                K += 1e-6 * np.eye(len(Y_norm))  # Noise for stability
                
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    K += 1e-4 * np.eye(len(Y_norm))
                    try:
                        L = np.linalg.cholesky(K)
                    except:
                        continue
                
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_norm))
                
                # We need to evaluate the acquisition function on a grid of candidate points.
                # To be efficient, we only evaluate EI on a subset of candidates.
                
                # Candidate points: Latin Hypercube + Local samples around best
                n_cand = 200
                # 50% global, 50% local
                n_global = n_cand // 2
                n_local = n_cand - n_global
                
                X_cand_global = rng.uniform(-5, 5, (n_global, dim))
                
                if self._best_x is not None:
                    X_cand_local = self._best_x + rng.normal(0, 1.0, (n_local, dim))
                    X_cand_local = np.clip(X_cand_local, -5, 5)
                else:
                    X_cand_local = rng.uniform(-5, 5, (n_local, dim))
                
                X_cand = np.vstack([X_cand_global, X_cand_local])
                
                # Compute K(X_cand, X_data)
                X_cand_sq = np.sum(X_cand ** 2, axis=1)
                D_cand = X_cand_sq[:, None] + X_sq[None, :] - 2 * X_cand @ X_data.T
                D_cand = np.maximum(D_cand, 0)
                K_star = np.exp(-D_cand / (2 * l ** 2))
                
                # Mean and Variance
                mu = K_star @ alpha
                v = np.linalg.solve(L, K_star.T)
                var = 1.0 - np.sum(v ** 2, axis=1)  # Since K(x,x)=1 for RBF
                var = np.maximum(var, 1e-6)
                std = np.sqrt(var)
                
                # Expected Improvement
                y_best = Y_norm.min()
                Z = (y_best - mu) / std
                ei = std * (norm.pdf(Z) + Z * norm.cdf(Z))
                
                # Normalize EI back to original scale? Not strictly necessary for selection, 
                # but let's check if we need to. Selection is based on ranking.
                
                # Add a bit of exploration noise to avoid getting stuck in one mode?
                # EI already balances this.
                
                # Select the best candidate
                idx = np.argmax(ei)
                x_next = X_cand[idx]
                
                # Score for kernel selection (max EI)
                score = ei[idx]
                
                if score > best_score:
                    best_score = score
                    best_mu = mu
                    best_cov = std
                    best_x_next = x_next
            
            if best_x_next is None:
                # Fallback: random
                best_x_next = rng.uniform(-5, 5, dim)
            
            return best_x_next, y_mean, y_std

        # 1. Initialization: Latin Hypercube
        n_init = 10
        X_init = rng.uniform(-5, 5, (n_init, dim))
        for i in range(n_init):
            if self._call_count >= budget:
                break
            if time.time() > deadline:
                break
            fx = wrapped(X_init[i])
            X.append(X_init[i])
            Y.append(fx)

        # 2. Iterative GP search
        while self._call_count < budget:
            if time.time() > deadline:
                break
            
            if len(X) < 2:
                # If we have very few points, just add random points
                x_next = rng.uniform(-5, 5, dim)
            else:
                try:
                    x_next, y_mean, y_std = fit_gp_and_acq(X, Y)
                except Exception:
                    # Fallback to random if GP fails
                    x_next = rng.uniform(-5, 5, dim)
            
            fx = wrapped(x_next)
            X.append(x_next)
            Y.append(fx)
            
            # Optional: If we have a lot of points, we might want to sub-sample to keep GP fast
            # But dim=5 and budget=500 is small enough that full GP should be fast enough.
            # 500 points in 5D is manageable.

        if self._best_x is None:
            self._best_x = np.zeros(dim)
            self._best_f = func(self._best_x)

        return self._best_x, self._best_f


# EVOLVE-BLOCK-END
