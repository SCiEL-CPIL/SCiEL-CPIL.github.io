"""Bayesian Optimization with Matern-5/2 GP and EI acquisition for BBOB f22."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Bayesian Optimizer for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        
        # State
        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0
        rng = np.random.default_rng(42)

        # Data storage for GP
        X_data = []
        y_data = []

        def do_call(x):
            nonlocal call_count, best_f, best_x
            if call_count >= budget or time.time() > deadline:
                return None
            x = np.clip(x, -5, 5)
            try:
                f = float(func(x))
            except Exception:
                f = np.inf
            call_count += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            X_data.append(x)
            y_data.append(f)
            return f

        # Phase 1: Initialization
        # Use a small set of diverse points to initialize the GP.
        # Stratified sampling helps cover the space quickly.
        n_init = 10
        init_points = []
        # Origin
        init_points.append(np.zeros(dim))
        # Random points in [-4, 4]
        for _ in range(n_init - 1):
            init_points.append(rng.uniform(-4, 4, size=dim))
        
        for p in init_points:
            f = do_call(p)
            if f is None:
                break

        if call_count >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 2: Bayesian Optimization Loop
        # We will iterate: 
        # 1. Fit GP to (X_data, y_data)
        # 2. Maximize EI over a candidate set
        # 3. Evaluate the best candidate
        
        # GP Parameters
        # Matern 5/2 kernel
        # y_scale: variance of y
        # length_scales: length scales for each dimension (isotropic for simplicity, or anisotropic)
        # noise: observation noise
        
        # Normalize y to [0, 1] or standardize? 
        # GP handles arbitrary scales, but EI is in units of y.
        # It's often better to standardize y for stable GP fitting.
        # However, EI is defined on the raw scale relative to best_f.
        # Let's standardize y for the GP fit, then transform back for EI calculation?
        # Or just fit on raw y. If y varies wildly, raw y might be unstable.
        # Let's standardize y for the GP model, but calculate EI on the raw scale.
        # Actually, standard EI formula:
        # EI(x) = (mu(x) - y_best - xi) * Phi(Z) + sigma(x) * phi(Z)
        # where Z = (mu(x) - y_best - xi) / sigma(x)
        # This works on raw scale.
        
        # To make GP fitting robust, we can standardize y_data before fitting.
        # Let's do that.
        
        max_iter = 400 # Max BO iterations
        
        for _ in range(max_iter):
            if call_count >= budget or time.time() > deadline:
                break
            
            n_samples = len(X_data)
            if n_samples < 5:
                # Need at least a few points for GP
                # Just do random sampling if too few
                x_next = rng.uniform(-5, 5, size=dim)
                do_call(x_next)
                continue

            X = np.array(X_data)
            y = np.array(y_data)
            
            # Standardize y
            y_mean = np.mean(y)
            y_std = np.std(y)
            if y_std < 1e-10:
                y_std = 1.0
            y_std = max(y_std, 1e-6)
            
            y_norm = (y - y_mean) / y_std
            
            # Fit GP
            # We need a GP implementation. SciPy doesn't have a built-in GP class that is easy to use for this.
            # We must implement a simple GP or use a library if available.
            # The prompt says "Python standard library, NumPy and SciPy are available."
            # No GPyTorch, No GPflow, No scikit-learn (GP module is in sklearn but not always recommended for this).
            # Wait, is sklearn available? "Python standard library, NumPy and SciPy". Usually implies no sklearn.
            # I will implement a minimal GP using NumPy.
            
            # Matern 5/2 Kernel:
            # k(x, x') = sigma^2 * (1 + sqrt(5)*d + 5/3*d^2) * exp(-sqrt(5)*d)
            # where d = ||x - x'|| / l (Euclidean distance / length scale)
            
            # Hyperparameters to optimize:
            # sigma^2 (signal variance), l (length scale), sigma_n^2 (noise variance)
            
            # For speed and simplicity in this constrained environment, 
            # we can use a fixed reasonable hyperparameter set or do a very quick grid search.
            # Given the 300s limit and 500 calls, fitting a GP 400 times must be fast.
            # NumPy based GP fit involves inverting an N x N matrix. N up to 500.
            # 500x500 inversion is fast (<1ms).
            # However, optimizing hyperparameters via MLE is expensive.
            # Let's use fixed hyperparameters tuned for this scale.
            # Domain is [-5, 5]^5. Distances up to ~14.
            # Let's try l = 1.0, sigma^2 = 1.0, noise = 1e-6.
            # Or better, adapt hyperparameters roughly?
            # Let's stick to fixed for now to ensure speed and stability.
            
            sigma2 = 1.0
            l = 1.0
            noise_var = 1e-6
            
            def matern52_kernel(X1, X2, l, sigma2):
                # X1: (N, D), X2: (M, D)
                # Returns (N, M)
                X1 = X1.reshape(-1, 1, dim)
                X2 = X2.reshape(1, -1, dim)
                # Squared distances
                sq_dists = np.sum((X1 - X2) ** 2, axis=2)
                d = np.sqrt(sq_dists + 1e-10) / l
                # Matern 5/2
                K = sigma2 * (1 + np.sqrt(5) * d + 5/3 * d**2) * np.exp(-np.sqrt(5) * d)
                return K
            
            K = matern52_kernel(X, X, l, sigma2)
            K += noise_var * np.eye(n_samples)
            
            # Cholesky decomposition for stability and speed
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # Fallback to random if GP fails
                x_next = rng.uniform(-5, 5, size=dim)
                do_call(x_next)
                continue
            
            # Solve for alpha: K alpha = y_norm
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_norm))
            
            # Acquisition Function: Expected Improvement
            # We need to maximize EI over the domain.
            # We can't maximize analytically in 5D easily.
            # Strategy: Generate a large batch of candidate points, evaluate EI, pick the best.
            # Or use a local optimizer on the EI function.
            # Given budget, a batch of 100-200 random candidates is cheap.
            
            n_candidates = 200
            X_cand = rng.uniform(-5, 5, size=(n_candidates, dim))
            
            # Predict mean and variance at candidates
            K_s = matern52_kernel(X_cand, X, l, sigma2) # (N_cand, N_data)
            
            # Mean: K_s @ alpha
            mu = K_s @ alpha
            # Transform back to raw scale
            mu_raw = mu * y_std + y_mean
            
            # Variance: sigma^2 - K_s @ K^{-1} @ K_s.T
            # K^{-1} = L^{-T} L^{-1}
            # v = L^{-1} K_s.T
            v = np.linalg.solve(L, K_s.T) # (N_data, N_cand)
            var = sigma2 - np.sum(v**2, axis=0) # (N_cand,)
            var = np.maximum(var, 1e-10)
            sigma = np.sqrt(var)
            
            # EI Calculation
            # y_best is best_f
            # xi is a small constant for exploration, e.g., 0.01 * range or 0.01
            xi = 0.01
            
            # Z = (mu_raw - best_f - xi) / sigma
            # Handle sigma == 0
            with np.errstate(divide='ignore', invalid='ignore'):
                Z = (mu_raw - best_f - xi) / sigma
                # EI = (mu_raw - best_f - xi) * Phi(Z) + sigma * phi(Z)
                # Phi is CDF of standard normal, phi is PDF
                # Use scipy.stats.norm if available? Yes, SciPy is available.
                from scipy.stats import norm
                EI = (mu_raw - best_f - xi) * norm.cdf(Z) + sigma * norm.pdf(Z)
                # Where sigma is 0, EI should be 0 if mu < best_f + xi, else (mu - best_f - xi)
                # norm.cdf(0) = 0.5, norm.pdf(0) = 0.3989
                # If sigma -> 0 and mu > best_f + xi, Z -> inf, cdf -> 1, pdf -> 0. EI -> mu - best_f - xi.
                # If sigma -> 0 and mu < best_f + xi, Z -> -inf, cdf -> 0, pdf -> 0. EI -> 0.
                # This is correct.
            
            # Select candidate with max EI
            idx_best = np.argmax(EI)
            x_next = X_cand[idx_best]
            
            # Evaluate
            f_next = do_call(x_next)
            if f_next is None:
                break

        # Phase 3: Polish
        # If budget remains, do local refinement around best_x using Nelder-Mead or random steps
        if call_count < budget and time.time() < deadline:
            remaining = budget - call_count
            # Use a simple local search
            step_size = 0.1
            for _ in range(min(remaining, 20)):
                if call_count >= budget or time.time() > deadline:
                    break
                # Random direction
                dir_vec = rng.normal(0, 1, size=dim)
                dir_vec /= np.linalg.norm(dir_vec)
                x_trial = best_x + step_size * dir_vec
                f_trial = do_call(x_trial)
                if f_trial is None:
                    break
                if f_trial < best_f:
                    step_size *= 0.5 # Shrink if improved
                else:
                    step_size *= 0.8 # Shrink slightly if not improved? Or keep?
                    # If not improved, maybe try a different direction next time.
                    # For now, just shrink to converge.
                    
        return best_x, best_f

# EVOLVE-BLOCK-END
