import numpy as np
import time
from scipy.optimize import minimize
import math

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, kill at 330s.
        # We reserve 20s for cleanup/return to be safe.
        deadline = start_time + 280
        
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        # Data structures for the GP
        X_obs = []
        y_obs = []
        
        best_x = None
        best_f = np.inf
        calls_used = 0
        
        def call_func(x):
            nonlocal calls_used, best_x, best_f
            if calls_used >= budget:
                raise Exception("Budget exceeded")
            if time.time() > deadline:
                raise Exception("Time limit exceeded")
            
            try:
                f_val = func(x)
            except Exception:
                # If the function call itself fails (unlikely per contract but safe), 
                # we treat it as a very bad value or stop.
                # Contract says "exception later keeps the calls made".
                # We should probably stop the loop if the underlying func is broken, 
                # but here we assume func is valid.
                raise
                
            calls_used += 1
            X_obs.append(x.copy())
            y_obs.append(f_val)
            
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
                
            return f_val

        def gp_predict(X_new):
            """
            Simple GP with RBF kernel.
            Returns mean and variance for X_new.
            """
            if len(X_obs) == 0:
                return np.zeros(len(X_new)), np.ones(len(X_new))
            
            X_obs_arr = np.array(X_obs)
            y_obs_arr = np.array(y_obs)
            
            # Standardize inputs for kernel stability
            # Using min-max scaling based on observed points is safer than fixed bounds 
            # if the search concentrates, but fixed bounds [-5,5] are known.
            # Let's use fixed bounds for simplicity and stability.
            X_min = np.full(n, -5.0)
            X_max = np.full(n, 5.0)
            
            X_obs_std = (X_obs_arr - X_min) / (X_max - X_min)
            X_new_std = (X_new - X_min) / (X_max - X_min)
            
            # Standardize outputs
            y_mean = np.mean(y_obs_arr)
            y_std = np.std(y_obs_arr)
            if y_std < 1e-8:
                y_std = 1.0
            y_obs_std = (y_obs_arr - y_mean) / y_std
            
            # RBF Kernel
            # Length scale l. For 5D Rosenbrock, the valley is narrow.
            # A smaller l captures local structure better.
            l = 0.2 
            alpha = 1e-2 # Noise
            
            def rbf_kernel(A, B, l):
                # Squared Euclidean distance
                sq_dists = np.sum(A**2, axis=1)[:, None] + np.sum(B**2, axis=1)[None, :] - 2 * np.dot(A, B.T)
                sq_dists = np.maximum(sq_dists, 0)
                return np.exp(-0.5 * sq_dists / (l**2))
            
            K = rbf_kernel(X_obs_std, X_obs_std, l)
            K[np.diag_indices_from(K)] += alpha
            
            K_s = rbf_kernel(X_obs_std, X_new_std, l)
            
            try:
                L = np.linalg.cholesky(K)
                alpha_vec = np.linalg.solve(L, y_obs_std)
                alpha_vec = np.linalg.solve(L.T, alpha_vec)
                
                mu_std = K_s.T @ alpha_vec
                v = np.linalg.solve(L, K_s)
                var_std = 1.0 - np.sum(v**2, axis=0)
                
                # Unstandardize
                mu = mu_std * y_std + y_mean
                var = var_std * y_std**2
                
                # Ensure variance is positive
                var = np.maximum(var, 1e-10)
                
            except np.linalg.LinAlgError:
                # Fallback if Cholesky fails (numerical instability)
                # Use simple heuristic: predict mean of obs, high variance
                mu = np.full(len(X_new), y_mean)
                var = np.full(len(X_new), y_std**2 + 1.0)
                
            return mu, var

        def expected_improvement(mu, var, best_f, xi=0.01):
            """
            Compute Expected Improvement for acquisition.
            xi is the exploration-exploitation trade-off.
            """
            var = np.maximum(var, 1e-10)
            sigma = np.sqrt(var)
            z = (best_f - mu - xi) / sigma
            ei = (best_f - mu - xi) * (1 - norm_cdf(z)) + sigma * norm_pdf(z)
            # If var is 0, EI is 0
            ei = np.where(sigma < 1e-8, 0.0, ei)
            return ei

        # Helper for Normal CDF and PDF (using math.erf for scalar, or vectorized approximation)
        # Since we are in a loop, we can use scipy.stats if available, but let's stick to numpy/math for speed
        # Actually, scipy.stats is available.
        from scipy.stats import norm
        
        # 1. Initial Sampling
        # We need enough points to build a decent GP.
        # 5D problem. 20-30 points is a reasonable start.
        n_init = 25
        # Mix of random and structured
        # Origin
        init_points = [np.zeros(n)]
        # Random uniform
        rand_points = rng.uniform(-5, 5, (n_init - 1, n))
        init_points.extend(rand_points)
        
        for x in init_points:
            try:
                call_func(x)
            except Exception:
                break
        
        if best_x is None:
            return np.zeros(n), np.inf

        # 2. GP-Based Global Search
        # We will alternate between GP acquisition and local polishing.
        # Strategy: 
        # - Use GP to find promising regions (EI acquisition).
        # - Once a good point is found, run a local optimizer (L-BFGS-B) to exploit it.
        # - Add the local optimizer's path to the GP? No, that's expensive. 
        # - Instead, add the local optimizer's result to the GP.
        # - Repeat until budget or time is low.
        
        # Number of GP iterations
        # We have ~475 calls left.
        # Each GP iteration costs 1 call for acquisition.
        # Each local polish might cost 50-100 calls.
        # Let's do ~10 rounds of (GP Pick + Local Polish).
        
        max_gp_rounds = 10
        local_budget_per_round = 40
        
        for i in range(max_gp_rounds):
            if calls_used >= budget - 10: # Keep 10 for safety
                break
            if time.time() > deadline - 10:
                break
                
            # Generate candidate points for acquisition
            # Sample 1000 points in the box
            n_candidates = 1000
            X_cand = rng.uniform(-5, 5, (n_candidates, n))
            
            # Predict mean and var
            mu, var = gp_predict(X_cand)
            
            # Compute EI
            ei = expected_improvement(mu, var, best_f, xi=0.05)
            
            # Select best candidate
            best_idx = np.argmax(ei)
            x_next = X_cand[best_idx]
            
            # Evaluate this point
            try:
                f_next = call_func(x_next)
            except Exception:
                break
            
            # If this point is significantly better, run local optimization
            # "Significantly" means it's the new best, or close to it
            if f_next <= best_f * 1.1 or (best_f > 0 and f_next < best_f):
                # Run L-BFGS-B
                remaining_budget = budget - calls_used
                # Cap local optimization to not eat all budget
                max_fev_local = min(remaining_budget, local_budget_per_round)
                
                if max_fev_local > 5:
                    def wrapper(x):
                        if time.time() > deadline:
                            raise ValueError("Time limit")
                        return call_func(x)
                    
                    try:
                        res = minimize(wrapper, x_next, method='L-BFGS-B',
                                       bounds=[(-5, 5)] * n,
                                       options={'maxiter': 100, 'maxfev': max_fev_local, 'ftol': 1e-10, 'gtol': 1e-8})
                        
                        if res.fun < best_f:
                            # Update best
                            best_f = res.fun
                            best_x = res.x.copy()
                            
                            # Add the result to the GP observation set
                            # Note: call_func already added the starting point x_next.
                            # The local optimizer made many calls which were added to X_obs/y_obs.
                            # This is good, it enriches the GP.
                            
                    except Exception:
                        # Local optimization failed or timed out
                        # We keep the best found so far
                        pass

            # If the GP point was not great, we just continue to the next GP iteration.
            # The GP model is updated with the new point (x_next, f_next).

        # 3. Final Polish if budget remains
        if calls_used < budget - 5 and time.time() < deadline - 5:
             # Try one more local optimization from the global best
             remaining = budget - calls_used
             max_fev = min(remaining, 50)
             
             def wrapper_final(x):
                 if time.time() > deadline:
                     raise ValueError("Time limit")
                 return call_func(x)
             
             try:
                 res = minimize(wrapper_final, best_x, method='L-BFGS-B',
                                bounds=[(-5, 5)] * n,
                                options={'maxiter': 100, 'maxfev': max_fev, 'ftol': 1e-12, 'gtol': 1e-10})
                 if res.fun < best_f:
                     best_f = res.fun
                     best_x = res.x.copy()
             except Exception:
                 pass

        return best_x, best_f

# Helper functions for Normal distribution (if scipy.stats is slow or unavailable, but it is available)
# We imported scipy.stats.norm inside the method to keep it local, but let's define global helpers for clarity/speed
from scipy.stats import norm
norm_cdf = norm.cdf
norm_pdf = norm.pdf
