import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import norm
import warnings

# Suppress warnings from scipy if convergence is not perfect, which is fine for our use case
warnings.filterwarnings("ignore")

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: The evaluator kills at 300s (or 330s in some contexts, 
        # but the prompt says 300s limit for evaluation, 330s kill). 
        # Let's be conservative: 280s deadline.
        deadline = start_time + 280.0
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        rng = np.random.default_rng(42)
        
        d = self.dim
        lb = np.full(d, -5.0)
        ub = np.full(d, 5.0)
        bounds = [(-5.0, 5.0)] * d

        def call_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
                
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, lb, ub)
            
            try:
                f = float(func(x))
                if not np.isfinite(f):
                    f = np.inf
            except Exception:
                f = np.inf
                
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Data storage for GP
        X_obs = []
        Y_obs = []

        def add_obs(x, y):
            X_obs.append(x)
            Y_obs.append(y)

        # Initial Sampling: Use a mix of random and grid-like points to cover space
        # 10-15 initial points are usually enough to start a GP in 5D.
        num_init = 15
        # Use a space-filling approach or just random? Random is fine for 5D with 15 pts.
        # To be more robust, let's use a simple Latin Hypercube or just random.
        # Random is simpler and less code.
        for _ in range(num_init):
            if calls_made >= self.budget or time.time() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, d)
            f = call_func(x)
            if f < np.inf:
                add_obs(x, f)

        if not X_obs:
            # Fallback if no valid points
            x = np.zeros(d)
            call_func(x)
            return best_x, best_f

        X_obs = np.array(X_obs)
        Y_obs = np.array(Y_obs)
        
        # GP Hyperparameters
        # We will use a simple GP implementation using standard linear algebra.
        # Kernel: Matern 5/2
        def matern52_kernel(X1, X2, length_scales, variance):
            X1 = np.atleast_2d(X1)
            X2 = np.atleast_2d(X2)
            # Normalize by length scales
            diff = (X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) / length_scales
            dist = np.sqrt(np.sum(diff ** 2, axis=2))
            sqrt_5 = np.sqrt(5.0)
            return variance * (1.0 + sqrt_5 * dist + 5.0 / 3.0 * dist ** 2) * np.exp(-sqrt_5 * dist)

        # We will optimize length scales and variance slightly, or use fixed reasonable ones.
        # Optimizing hyperparameters at every step is expensive. 
        # Strategy: Optimize hyperparameters once after initial sampling, then keep them fixed 
        # or update them every few steps if time permits.
        # For simplicity and speed, let's use fixed, reasonable hyperparameters initially.
        # If the function is very noisy or has specific scales, this might hurt.
        # Better: Optimize log-likelihood of GP every ~10-20 steps.
        
        # Initial guess for hyperparameters
        # Length scales: start around 1.0
        # Variance: start around 1.0 (or scaled to data)
        
        # To make it robust, let's normalize the output Y for GP training.
        # GP works better with zero mean, unit variance.
        
        def fit_gp_hyperparams(X, Y):
            # Optimize log_marginal_likelihood
            # Params: log_length_scales (d), log_variance, log_noise
            # We fix noise to 1e-8 to avoid singular matrices if data is deterministic
            
            d = X.shape[1]
            initial_log_ls = np.zeros(d)
            initial_log_var = np.log(np.var(Y) + 1e-6)
            log_noise = np.log(1e-8)
            
            def neg_log_likelihood(params):
                log_ls = params[:d]
                log_var = params[d]
                # log_noise is fixed
                ls = np.exp(log_ls)
                var = np.exp(log_var)
                
                K = matern52_kernel(X, X, ls, var)
                # Add noise
                K[np.diag_indices_from(K)] += 1e-8
                
                try:
                    L = np.linalg.cholesky(K)
                except np.linalg.LinAlgError:
                    return 1e10
                    
                # Log likelihood: -0.5 * y^T K^-1 y - 0.5 * log(det(K)) - d/2 * log(2pi)
                # log(det(K)) = 2 * sum(log(diag(L)))
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
                mll = -0.5 * np.dot(Y, alpha) - 0.5 * log_det
                return -mll
            
            # Use L-BFGS-B to maximize MLL (minimize neg MLL)
            # Bounds for log_ls: [-2, 2] (ls in [0.13, 7.4])
            # Bounds for log_var: [-2, 2]
            bounds_opt = [(-2.0, 2.0)] * d + [(-2.0, 2.0)]
            x0 = np.concatenate([initial_log_ls, [initial_log_var]])
            
            try:
                res = minimize(neg_log_likelihood, x0, method='L-BFGS-B', bounds=bounds_opt, options={'maxiter': 50})
                if res.success:
                    log_ls = res.x[:d]
                    log_var = res.x[d]
                    return np.exp(log_ls), np.exp(log_var)
                else:
                    # Fallback to initial
                    return np.exp(initial_log_ls), np.exp(initial_log_var)
            except Exception:
                return np.exp(initial_log_ls), np.exp(initial_log_var)

        # Fit initial hyperparameters
        length_scales, variance = fit_gp_hyperparams(X_obs, Y_obs)
        
        # Store for reuse
        current_ls = length_scales
        current_var = variance

        # Main BO Loop
        # We will run BO until we have used, say, 80% of the budget.
        # Then use the remaining 20% for local polish.
        bo_budget_limit = int(self.budget * 0.8)
        
        # Acquisition Function: Expected Improvement
        def expected_improvement(x, X_train, Y_train, ls, var):
            # Predict mean and std at x
            x = np.atleast_2d(x)
            K_x = matern52_kernel(x, X_train, ls, var)
            K = matern52_kernel(X_train, X_train, ls, var)
            K[np.diag_indices_from(K)] += 1e-8
            
            try:
                L = np.linalg.cholesky(K)
            except np.linalg.LinAlgError:
                return 0.0
                
            v = np.linalg.solve(L.T, np.linalg.solve(L, K_x.T))
            
            # Mean prediction
            # Note: We should ideally center Y, but for EI we need the actual best value.
            # Let's just use raw Y.
            mu = np.dot(v.T, Y_train)
            
            # Variance prediction
            # Var(x) = K(x,x) - K(x,X) K(X,X)^-1 K(X,x)
            # K(x,x) is var (diagonal of kernel with same point is var)
            # Actually, matern52_kernel(x,x) returns var * (1+0+0)*1 = var.
            v_var = var - np.sum(v ** 2)
            if v_var < 0:
                v_var = 1e-8
            std = np.sqrt(v_var)
            
            best_y = np.min(Y_train)
            
            if std < 1e-8:
                return 0.0
                
            z = (best_y - mu) / std
            EI = (best_y - mu) * norm.cdf(z) + std * norm.pdf(z)
            return EI[0]

        # Optimize EI to find next point
        def maximize_ei(X_train, Y_train, ls, var):
            # We use multiple random restarts to maximize EI
            # EI is non-convex
            best_ei_val = -np.inf
            best_x = None
            
            # Number of restarts: 10-20 is usually good for 5D
            n_restarts = 15
            
            for _ in range(n_restarts):
                x0 = rng.uniform(-5.0, 5.0, d)
                
                def neg_ei(x):
                    return -expected_improvement(x, X_train, Y_train, ls, var)
                
                try:
                    res = minimize(neg_ei, x0, method='L-BFGS-B', bounds=bounds, options={'maxiter': 50, 'ftol': 1e-8})
                    if res.success:
                        ei_val = -res.fun
                        if ei_val > best_ei_val:
                            best_ei_val = ei_val
                            best_x = res.x
                except Exception:
                    continue
            
            if best_x is None:
                # Fallback to random
                best_x = rng.uniform(-5.0, 5.0, d)
                
            return best_x

        # Loop
        while calls_made < bo_budget_limit and time.time() < deadline:
            # Check if we need to re-fit hyperparameters
            # Re-fitting every 10-20 points is a good balance
            if len(X_obs) % 15 == 0 and len(X_obs) > num_init:
                if time.time() < deadline - 5.0: # Reserve 5s for safety
                    current_ls, current_var = fit_gp_hyperparams(np.array(X_obs), np.array(Y_obs))
            
            # Find next point using EI
            x_next = maximize_ei(np.array(X_obs), np.array(Y_obs), current_ls, current_var)
            
            # Evaluate
            f_next = call_func(x_next)
            
            if f_next < np.inf:
                add_obs(x_next, f_next)
            
            # If we have very few calls left, break early to save time for polish
            if calls_made > self.budget - 50:
                break

        # Phase 3: Local Polish
        # Use remaining budget to polish the best point found by BO
        if calls_made < self.budget and time.time() < deadline and best_x is not None:
            remaining = self.budget - calls_made
            if remaining > 10:
                # L-BFGS-B for smooth convergence
                try:
                    res = minimize(
                        call_func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=bounds,
                        options={
                            'maxfev': min(50, remaining),
                            'ftol': 1e-12,
                            'gtol': 1e-8
                        }
                    )
                except Exception:
                    pass
                
                # Nelder-Mead for robustness
                if calls_made < self.budget and time.time() < deadline:
                    remaining = self.budget - calls_made
                    if remaining > 5:
                        try:
                            minimize(
                                call_func,
                                best_x,
                                method='Nelder-Mead',
                                options={
                                    'maxfev': min(20, remaining),
                                    'xatol': 1e-8,
                                    'fatol': 1e-8
                                }
                            )
                        except Exception:
                            pass

        if best_x is None:
            best_x = np.zeros(d)
            call_func(best_x)

        return best_x, best_f
