import numpy as np
import time
from scipy.stats import norm
from scipy.spatial.distance import cdist
import warnings

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        time_limit = 325.0 # 5s margin before 330s kill

        # BO setup
        # Initial sampling: Use a mix of random and grid-like points to seed the GP
        n_init = min(20, budget // 10)
        
        # Deterministic initial points for reproducibility and coverage
        # We want to cover the space [-5, 5]^5
        # A simple stratified sampling or Latin Hypercube is better than pure random for BO initialization
        # Since we can't use scipy.stats.qmc easily without imports (standard lib + numpy + scipy allowed), 
        # we can generate a simple LHS using numpy permutations or just use a fixed set of well-spread points.
        
        # Let's use a fixed set of points based on a simple grid + noise for the first few, 
        # or just deterministic random with a fixed seed for the initial batch.
        # The task says "hidden seeded shift", but the optimizer's internal randomness 
        # should be controlled. Using a fixed seed for the optimizer's internal RNG 
        # ensures deterministic behavior for the same function instance.
        
        rng = np.random.default_rng(42)
        
        X = np.zeros((0, dim))
        Y = np.zeros(0)
        
        best_x = None
        best_f = np.inf
        calls = 0

        # Phase 1: Initial Exploration
        # Generate n_init points. 
        # To ensure coverage, we can use a simple space-filling approach.
        # For dim=5, a full grid is too large. 
        # Let's use a mix: center, corners, and random.
        
        # Simple deterministic initialization
        init_points = []
        # Center
        init_points.append(np.zeros(dim))
        # Random points with fixed seed
        for _ in range(n_init - 1):
            init_points.append(rng.uniform(-5, 5, dim))
        
        X_init = np.array(init_points)
        
        for i in range(n_init):
            if time.time() - start_time > time_limit:
                break
            x = X_init[i]
            f = func(x)
            calls += 1
            X = np.vstack([X, x]) if X.size > 0 else x.reshape(1, dim)
            Y = np.append(Y, f)
            
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Phase 2: Bayesian Optimization Loop
        # Hyperparameters for GP
        # Matern 5/2 kernel is good for smooth-ish functions, RBF for smoother ones.
        # BBOB functions like Rastrigin are not very smooth, but GP can still help guide.
        # We will use a simple RBF kernel with a fixed length scale initially, 
        # or attempt a very simple hyperparameter optimization if time permits.
        # Given 300 calls and dim 5, hyperparameter optimization is expensive.
        # We will use a fixed, reasonably tuned kernel.
        
        # Kernel: RBF (Gaussian)
        # k(x, x') = sigma_f^2 * exp(-0.5 * ||x - x'||^2 / l^2)
        # Noise: sigma_n^2
        
        # Heuristic hyperparameters:
        # sigma_f: 1.0 (scale of function values)
        # l: 1.5 (length scale, allows seeing structure over ~1-2 units)
        # sigma_n: 0.1 (noise)
        
        sigma_f = 1.0
        l = 1.5
        sigma_n = 0.05
        
        # Standardize Y for numerical stability in GP
        Y_mean = np.mean(Y)
        Y_std = np.std(Y)
        if Y_std < 1e-8:
            Y_std = 1.0
            
        Y_norm = (Y - Y_mean) / Y_std
        
        # Function to compute GP predictions
        def gp_predict(X_new, X_train, Y_train_norm, sigma_f, l, sigma_n):
            """
            Returns mean and std of GP at X_new
            """
            # Compute kernel matrix K(X_train, X_train)
            # cdist computes Euclidean distance
            d = cdist(X_train, X_train)
            K = sigma_f**2 * np.exp(-0.5 * (d**2) / (l**2))
            K += sigma_n**2 * np.eye(len(X_train))
            
            # Compute kernel vector k(X_new, X_train)
            d_new = cdist(X_new, X_train)
            k_new = sigma_f**2 * np.exp(-0.5 * (d_new**2) / (l**2))
            
            # Solve for alpha
            # K alpha = Y_train_norm
            try:
                alpha = np.linalg.solve(K, Y_train_norm)
            except np.linalg.LinAlgError:
                # Fallback to least squares if singular
                alpha = np.linalg.lstsq(K, Y_train_norm, rcond=None)[0]
                
            # Mean prediction: k_new @ alpha
            mu = k_new @ alpha
            
            # Variance prediction: sigma_f^2 - k_new @ K^-1 @ k_new.T
            # v = K^-1 @ k_new.T
            try:
                v = np.linalg.solve(K, k_new.T)
            except np.linalg.LinAlgError:
                v = np.linalg.lstsq(K, k_new.T, rcond=None)[0]
                
            var = sigma_f**2 - np.sum(k_new * v, axis=1)
            var = np.maximum(var, 1e-8) # Numerical stability
            
            # Unstandardize
            mu_unnorm = mu * Y_std + Y_mean
            std_unnorm = np.sqrt(var) * Y_std
            
            return mu_unnorm, std_unnorm

        # Expected Improvement (EI) acquisition function
        def expected_improvement(mu, sigma, best_f_current):
            # EI = (best_f - mu) * Phi((best_f - mu)/sigma) + sigma * phi((best_f - mu)/sigma)
            # where Phi is CDF, phi is PDF of standard normal
            # Note: We are minimizing.
            # Improvement is defined as best_f - f(x)
            # If mu < best_f, EI is positive.
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                z = (best_f_current - mu) / (sigma + 1e-8)
                ei = (best_f_current - mu) * norm.cdf(z) + sigma * norm.pdf(z)
                # Handle case where sigma is very small
                ei = np.where(sigma < 1e-6, 0.0, ei)
            return ei

        # Candidate generation strategy:
        # Generate a batch of candidates, evaluate EI, pick the best.
        # To keep it fast, we don't use a massive grid every time.
        # We can use random candidates around current best and global randoms.
        
        # We will maintain a "candidate pool" or generate on the fly.
        # Given dim=5, generating 100-200 random candidates per step is cheap.
        
        n_candidates_per_step = 200
        
        while calls < budget:
            if time.time() - start_time > time_limit:
                break
            
            # Generate candidates
            # Mix of:
            # 1. Global random (exploration)
            # 2. Local random around best_x (exploitation)
            # 3. Local random around current GP mean argmin (if available, but we just use best_x for simplicity)
            
            # Let's just use global randoms for simplicity and robustness, 
            # as GP guides the selection among them.
            # Or, better: 
            # 50% global uniform
            # 50% local around best_x with sigma decreasing over time
            
            progress = calls / budget
            local_sigma = 2.0 * (1.0 - progress) + 0.5 # Decreases from 2.5 to 0.5
            
            n_global = n_candidates_per_step // 2
            n_local = n_candidates_per_step - n_global
            
            X_global = rng.uniform(-5, 5, (n_global, dim))
            X_local = best_x + rng.normal(0, local_sigma, (n_local, dim))
            X_local = np.clip(X_local, -5, 5)
            
            X_cands = np.vstack([X_global, X_local])
            
            # Evaluate GP
            try:
                mu, sigma = gp_predict(X_cands, X, Y_norm, sigma_f, l, sigma_n)
            except Exception:
                # If GP fails, fallback to random
                x_next = rng.uniform(-5, 5, dim)
                f_next = func(x_next)
                calls += 1
                X = np.vstack([X, x_next])
                Y = np.append(Y, f_next)
                if f_next < best_f:
                    best_f = f_next
                    best_x = x_next.copy()
                continue
            
            # Calculate EI
            ei = expected_improvement(mu, sigma, best_f)
            
            # Select candidate with max EI
            idx = np.argmax(ei)
            x_next = X_cands[idx]
            
            # Evaluate function
            f_next = func(x_next)
            calls += 1
            
            # Update data
            X = np.vstack([X, x_next])
            Y = np.append(Y, f_next)
            
            if f_next < best_f:
                best_f = f_next
                best_x = x_next.copy()
                
            # Update Y_norm
            Y_mean = np.mean(Y)
            Y_std = np.std(Y)
            if Y_std < 1e-8:
                Y_std = 1.0
            Y_norm = (Y - Y_mean) / Y_std
            
            # Optional: Simple hyperparameter adaptation
            # If we stagnate, maybe increase length scale to explore more?
            # For now, keep fixed.
            
        return best_x, best_f
