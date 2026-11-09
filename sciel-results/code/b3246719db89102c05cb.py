"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import random

class Optimizer:
    """Bayesian Optimizer with Gaussian Process surrogate for BBOB f15."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # Initial random sampling to build initial model
        # For dim=5, 10-15 initial points is reasonable
        n_init = min(15, budget)
        
        X = []
        Y = []
        
        # Ensure we have at least one point to start
        # We will use a fixed seed for reproducibility of the random part, 
        # but the function itself is black-box.
        rng = np.random.default_rng(42)
        
        for _ in range(n_init):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            y = func(x)
            X.append(x)
            Y.append(y)
            
        if not X:
            # Fallback if we couldn't even make one call
            return np.zeros(dim), 0.0

        X = np.array(X)
        Y = np.array(Y)
        
        best_x = X[np.argmin(Y)]
        best_y = np.min(Y)
        calls_made = len(X)
        
        # GP Parameters
        # We use a simple RBF kernel with standardization
        # For speed and stability in 5D, we might need to be careful with the length scale
        # Let's implement a simple GP with manual matrix inversion or use a library if available?
        # Standard library + NumPy + SciPy are available.
        # We'll implement a lightweight GP to avoid heavy dependencies or slow convergence of hyperparameter optimization.
        
        def gp_predict(X_train, Y_train, X_test):
            """
            Simple GP with RBF kernel.
            Hyperparameters are fixed for speed/stability in this context, 
            or we could do a very light optimization.
            Given the small budget, fixed reasonable hyperparameters might be better 
            than spending calls/time on hyperparameter tuning.
            """
            # Standardize inputs and outputs for numerical stability
            X_mean = np.mean(X_train, axis=0)
            X_std = np.std(X_train, axis=0) + 1e-8
            Y_mean = np.mean(Y_train)
            Y_std = np.std(Y_train) + 1e-8
            
            X_train_std = (X_train - X_mean) / X_std
            Y_train_std = (Y_train - Y_mean) / Y_std
            
            # Kernel parameters
            # Length scale: start with 1.0 in standardized space
            l = 1.0
            # Signal variance
            sigma_f = 1.0
            # Noise variance
            sigma_n = 1e-4
            
            def kernel(X1, X2, l, sigma_f):
                # RBF Kernel
                # dist^2 = sum((x1-x2)^2)
                X1_sq = np.sum(X1**2, axis=1)[:, np.newaxis]
                X2_sq = np.sum(X2**2, axis=1)[np.newaxis, :]
                dist_sq = np.maximum(X1_sq + X2_sq - 2 * X1 @ X2.T, 0)
                K = sigma_f**2 * np.exp(-0.5 * dist_sq / l**2)
                return K
            
            K = kernel(X_train_std, X_train_std, l, sigma_f)
            K += sigma_n**2 * np.eye(len(X_train))
            
            # Solve for alpha
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(np.linalg.cholesky(K).T, np.linalg.solve(L, Y_train_std))
            except np.linalg.LinAlgError:
                # Fallback to pinv if not positive definite
                alpha = np.linalg.pinv(K) @ Y_train_std
                L = np.eye(len(X_train)) # Not used in this branch for prediction mean, but needed for var?
                # Actually, if Cholesky fails, prediction variance is tricky. 
                # Let's add jitter and retry once.
                K += 1e-6 * np.eye(len(X_train))
                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(np.linalg.cholesky(K).T, np.linalg.solve(L, Y_train_std))
                except:
                    # Last resort: return mean only, zero variance
                    return np.full(X_test.shape[0], Y_mean), np.zeros(X_test.shape[0])

            # Predict
            X_test_std = (X_test - X_mean) / X_std
            K_star = kernel(X_test_std, X_train_std, l, sigma_f)
            
            mu = K_star @ alpha
            mu = mu * Y_std + Y_mean
            
            # Variance
            v = np.linalg.solve(L, K_star.T)
            var = sigma_f**2 - np.sum(K_star * v.T, axis=1)
            var = np.maximum(var, 1e-8)
            std = np.sqrt(var) * Y_std
            
            return mu, std

        def expected_improvement(mu, sigma, best_y, xi=0.01):
            """
            Calculate Expected Improvement.
            mu, sigma: arrays
            best_y: current best value
            """
            imp = best_y - mu - xi
            z = imp / (sigma + 1e-8)
            # Standard normal CDF and PDF
            # Using scipy if available, otherwise manual approximation
            try:
                from scipy.stats import norm
                ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
            except ImportError:
                # Manual approximation for norm.cdf and norm.pdf
                # norm.pdf(z) = exp(-z^2/2) / sqrt(2*pi)
                pdf = np.exp(-0.5 * z**2) / np.sqrt(2 * np.pi)
                # norm.cdf(z) = 0.5 * (1 + erf(z/sqrt(2)))
                # Simple erf approximation
                def erf_approx(x):
                    # Abramowitz and Stegun 7.1.26
                    sign = np.sign(x)
                    x = np.abs(x)
                    t = 1.0 / (1.0 + 0.3275911 * x)
                    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-x*x)
                    return sign * y
                cdf = 0.5 * (1.0 + erf_approx(z / np.sqrt(2.0)))
                ei = imp * cdf + sigma * pdf
            
            # Ensure EI is non-negative
            ei = np.maximum(ei, 0.0)
            return ei

        # Main Optimization Loop
        while calls_made < budget:
            if time.time() > deadline:
                break
            
            # Check for stagnation: if we haven't improved in the last 20 calls, restart or explore more
            # For now, let's just proceed. If GP variance is high, it will explore.
            
            # Select next point
            # We need to balance exploration and exploitation.
            # In BO, we typically maximize EI.
            # However, with a fixed GP, we might get stuck in local optima if the kernel is too smooth.
            # Let's add a small random component or use a "virtual" population.
            
            # Strategy: Generate a set of candidate points, evaluate their EI, pick the best.
            # How many candidates? 100-200 is usually fast enough for dim=5.
            n_candidates = 200
            
            # Sample candidates from a distribution that covers the space
            # We can sample uniformly, or around the current best.
            # Let's sample uniformly in the box to ensure global coverage, 
            # but weight them towards the best region? 
            # Actually, EI naturally handles this if the GP is good.
            # But if the GP is bad (early on), uniform is safer.
            
            # Let's generate candidates
            candidates = rng.uniform(-5, 5, (n_candidates, dim))
            
            # Predict
            mu, std = gp_predict(X, Y, candidates)
            
            # Calculate EI
            ei = expected_improvement(mu, std, best_y)
            
            # Pick the candidate with highest EI
            idx = np.argmax(ei)
            next_x = candidates[idx]
            
            # Evaluate
            next_y = func(next_x)
            calls_made += 1
            
            X = np.vstack([X, next_x])
            Y = np.append(Y, next_y)
            
            if next_y < best_y:
                best_y = next_y
                best_x = next_x.copy()
                
            # Optional: If we are stuck, we might want to force some exploration.
            # But EI with GP should do this if uncertainty is high.
            
        return best_x, best_y

# EVOLVE-BLOCK-END
