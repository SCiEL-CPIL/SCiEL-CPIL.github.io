"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    """Sequential Bayesian Optimizer with Matern 5/2 GP and EI acquisition."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Hyperparameters for GP
        # Initial length scales: assume features are on similar scales [-5, 5]
        # We will update these based on data
        l0 = 2.0
        sigma_f0 = 1.0
        
        X = []
        Y = []
        
        best_x = None
        best_f = float('inf')
        used_calls = 0
        
        # Helper to check time
        def time_ok():
            return time.time() < deadline and used_calls < budget

        # Helper to evaluate a point
        def eval_point(x):
            nonlocal used_calls, best_x, best_f
            x = np.clip(x, -5, 5)
            f = func(x)
            used_calls += 1
            X.append(x.copy())
            Y.append(f)
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # 1. Initial Sampling: Random points to establish initial GP
        # Use a spread out set of points
        n_init = min(10, budget)
        # Use a quasi-random sequence for better initial coverage
        # Simple Halton-like or just random with rejection? Random is fine for n=10 in 5D
        for _ in range(n_init):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, dim)
            eval_point(x)
            
        # 2. Main Loop: Sequential BO
        # We will fit a GP and pick the next point using EI
        # To keep it fast, we will limit the number of points used in the GP model
        # and use a simplified hyperparameter optimization or fixed/adaptive hyperparams.
        # Given 500 calls and 5D, fitting a full GP with hyperopt every step might be slow.
        # Strategy: 
        # - Use a subset of points if N gets large? No, 500 points in 5D is manageable for GP if we don't do heavy hyperopt.
        # - We will do a light hyperparameter optimization: 
        #   1. Initial fit with default params.
        #   2. Every 10 steps, do a quick grid search or L-BFGS on log-likelihood.
        #   3. Alternatively, just use adaptive length scales based on distance between best points?
        #   Let's stick to a simplified approach: 
        #   Fit GP with fixed noise, and optimize length scales and signal variance using a simple 
        #   gradient-based method on a subset of points if N > 20, or all points if N <= 20.
        
        # Actually, for robustness and speed, let's use a simpler surrogate model approach 
        # if GP fitting becomes too slow. But the prompt asks for GP.
        # Let's implement a lightweight GP updater.
        
        # GP Model Implementation
        def matern52_kernel(X1, X2, length_scales, sigma_f):
            # X1: (n1, d), X2: (n2, d)
            # length_scales: (d,)
            # sigma_f: scalar
            X1_scaled = X1 / length_scales
            X2_scaled = X2 / length_scales
            # Squared distance
            sq_dists = np.sum(X1_scaled**2, axis=1)[:, None] + np.sum(X2_scaled**2, axis=1)[None, :] - 2 * X1_scaled @ X2_scaled.T
            sq_dists = np.maximum(sq_dists, 0)
            r = np.sqrt(sq_dists)
            kernel = sigma_f**2 * (1 + np.sqrt(5) * r + 5/3 * r**2) * np.exp(-np.sqrt(5) * r)
            return kernel

        def log_likelihood(params, X, Y, noise_var=1e-4):
            # params: [log(length_scales...), log(sigma_f)]
            log_ls = params[:dim]
            log_sf = params[dim]
            ls = np.exp(log_ls)
            sf = np.exp(log_sf)
            
            K = matern52_kernel(X, X, ls, sf)
            K += noise_var * np.eye(len(X))
            
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                # Add jitter if singular
                K += 1e-6 * np.eye(len(X))
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    return -1e10 # Penalize
            
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))
            log_lik = -0.5 * Y @ alpha - np.sum(np.log(np.diag(L))) - 0.5 * len(Y) * np.log(2 * np.pi)
            return log_lik

        def predict_gp(X_new, X_train, Y_train, ls, sf, noise_var=1e-4):
            K = matern52_kernel(X_train, X_train, ls, sf)
            K += noise_var * np.eye(len(X_train))
            K_s = matern52_kernel(X_train, X_new, ls, sf)
            
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                K += 1e-6 * np.eye(len(X_train))
                L = np.linalg.cholesky(K)
                
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y_train))
            
            mu = K_s.T @ alpha
            v = np.linalg.solve(L, K_s)
            var = sf**2 - np.sum(v**2, axis=0)
            var = np.maximum(var, 1e-10)
            std = np.sqrt(var)
            
            return mu, std

        # Current best hyperparameters (log space)
        current_params = np.array([np.log(l0)] * dim + [np.log(sigma_f0)])
        
        # We will re-estimate hyperparameters periodically
        last_hyper_fit = 0
        
        for step in range(used_calls, budget):
            if not time_ok():
                break
            
            # Re-fit hyperparameters every 20 steps or if N is small
            if len(X) <= 20 or (step - last_hyper_fit) >= 20:
                # Optimize hyperparameters
                # Use a simple optimization. Since dim=5, we have 6 params.
                # We can use scipy.optimize.minimize on negative log-likelihood
                # To speed up, we can use a subset of points if N is large? 
                # For N=500, Cholesky is O(N^3) ~ 125M ops. Might be slow in Python loops.
                # Let's limit the GP training set to the most recent 100 points + best points?
                # Or just accept the cost. 300s / 500 steps = 0.6s per step.
                # Cholesky of 500x500 is fast in BLAS. The bottleneck is the Python overhead of the kernel computation.
                # Let's try to keep it efficient.
                
                def neg_ll(params):
                    return -log_likelihood(params, np.array(X), np.array(Y))
                
                # Optimization
                try:
                    res = minimize(neg_ll, current_params, method='L-BFGS-B', options={'maxiter': 50})
                    if res.success:
                        current_params = res.x
                    last_hyper_fit = step
                except Exception:
                    pass # Keep old params if optimization fails
            
            ls = np.exp(current_params[:dim])
            sf = np.exp(current_params[dim])
            
            # Generate candidates for EI
            # Sample points from a proposal distribution. 
            # A good proposal is a Gaussian centered at the current best, with a width that decays slightly.
            # Or use a grid? In 5D, grid is bad. 
            # Let's use a mixture: 
            # 1. Local search around best point
            # 2. Random points in the box (for exploration)
            
            n_candidates = 200
            # 80% local, 20% global
            n_local = 150
            n_global = 50
            
            candidates = []
            
            # Local candidates: Gaussian around best_x
            # Scale: start with 2.0, decay to 0.1
            progress = step / budget
            local_scale = 2.0 * (1 - progress) + 0.1
            for _ in range(n_local):
                x_cand = best_x + local_scale * rng.standard_normal(dim)
                x_cand = np.clip(x_cand, -5, 5)
                candidates.append(x_cand)
            
            # Global candidates: Uniform
            for _ in range(n_global):
                x_cand = rng.uniform(-5, 5, dim)
                candidates.append(x_cand)
                
            candidates = np.array(candidates)
            
            # Predict on candidates
            mu, std = predict_gp(candidates, np.array(X), np.array(Y), ls, sf)
            
            # Calculate Expected Improvement (EI)
            # EI = (f_best - mu) * Phi((f_best - mu) / std) + std * phi((f_best - mu) / std)
            # Note: We are minimizing. So f_best is the minimum found so far.
            # Standard EI for minimization:
            # Z = (f_best - mu) / std
            # EI = (f_best - mu) * Phi(Z) + std * phi(Z)
            
            Z = (best_f - mu) / std
            ei = (best_f - mu) * norm.cdf(Z) + std * norm.pdf(Z)
            
            # Select best candidate
            idx = np.argmax(ei)
            next_x = candidates[idx]
            
            # Evaluate
            eval_point(next_x)
            
        return best_x, best_f

# EVOLVE-BLOCK-END
