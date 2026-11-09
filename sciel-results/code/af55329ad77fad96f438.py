import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # 10s safety margin
        
        dim = self.dim
        total_budget = self.budget
        
        # State variables
        X_obs = []  # List of points evaluated
        Y_obs = []  # List of function values
        best_x = np.zeros(dim)
        best_f = np.inf
        calls_used = 0
        
        # Helper to evaluate function and update state
        def evaluate(x):
            nonlocal calls_used, best_x, best_f
            if calls_used >= total_budget:
                return np.inf
            x = np.clip(x, -5, 5)
            calls_used += 1
            try:
                f = float(func(x))
                if np.isfinite(f):
                    X_obs.append(x.copy())
                    Y_obs.append(f)
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                    return f
                else:
                    return np.inf
            except Exception:
                return np.inf

        # 1. Initial Sampling: Latin Hypercube for diverse coverage
        # Use 20 points for initial GP seeding
        n_init = min(20, total_budget)
        rng = np.random.default_rng(42)
        
        # Generate LHS points
        # For each dimension, divide [0, 1] into n_init intervals, pick one random point in each, then permute
        lhs_points = np.zeros((n_init, dim))
        for d in range(dim):
            # Intervals
            intervals = np.linspace(0, 1, n_init + 1)
            # Random point in each interval
            points_in_intervals = rng.uniform(intervals[:-1], intervals[1:])
            # Permute to decorrelate dimensions
            perm = rng.permutation(n_init)
            lhs_points[:, d] = points_in_intervals[perm]
        
        # Scale to [-5, 5]
        lhs_points = lhs_points * 10.0 - 5.0
        
        for i in range(n_init):
            if time.time() > deadline:
                break
            evaluate(lhs_points[i])
        
        # 2. Surrogate-Assisted Loop
        # We will use the remaining budget for GP-guided search
        # GP parameters
        # Matern 5/2 kernel: k(r) = (1 + sqrt(5)*r + 5/3*r^2) * exp(-sqrt(5)*r)
        # Scale parameters: length_scale, signal_variance, noise_variance
        
        # Fixed hyperparameters for speed and stability in 5D
        # Length scale around 1.0-2.0 for [-5,5] range
        length_scale = 1.5
        signal_var = 10.0
        noise_var = 1e-6
        
        def matern52_kernel(X1, X2, l, sig):
            """
            X1: (n1, d)
            X2: (n2, d)
            Returns: (n1, n2) kernel matrix
            """
            # Compute squared distances
            X1_norm = np.sum(X1**2, axis=1)[:, np.newaxis]
            X2_norm = np.sum(X2**2, axis=1)[np.newaxis, :]
            sq_dists = X1_norm + X2_norm - 2 * X1 @ X2.T
            sq_dists = np.maximum(sq_dists, 0)
            dists = np.sqrt(sq_dists)
            r = dists / l
            return sig**2 * (1 + np.sqrt(5)*r + 5/3*r**2) * np.exp(-np.sqrt(5)*r)

        def fit_gp(X, Y, l, sig, noise):
            """
            Fit GP to data. Returns K, L (Cholesky), alpha.
            X: (n, d), Y: (n,)
            """
            n = len(Y)
            K = matern52_kernel(X, X, l, sig)
            K[np.diag_indices(n)] += noise
            try:
                L = np.linalg.cholesky(K + 1e-8 * np.eye(n))
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))
                return K, L, alpha
            except np.linalg.LinAlgError:
                # Add jitter
                K += 1e-6 * np.eye(n)
                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, Y))
                    return K, L, alpha
                except np.linalg.LinAlgError:
                    return None, None, None

        def gp_predict(X_new, X, Y, L, alpha, l, sig, noise):
            """
            Predict mean and variance at X_new.
            X_new: (m, d)
            Returns: mean (m,), var (m,)
            """
            if L is None:
                # Fallback if GP failed: return mean and high variance
                mean = np.mean(Y) if len(Y) > 0 else 0
                var = np.var(Y) if len(Y) > 1 else 1
                return np.full(len(X_new), mean), np.full(len(X_new), var)
            
            K_star = matern52_kernel(X_new, X, l, sig)
            mean = K_star @ alpha
            v = np.linalg.solve(L, K_star.T)
            K_ss = matern52_kernel(X_new, X_new, l, sig)
            var = K_ss[np.diag_indices(len(X_new))] - np.sum(v**2, axis=0)
            var = np.maximum(var, 1e-8)
            return mean, var

        def expected_improvement(x_new, X, Y, L, alpha, l, sig, noise, best_f_current):
            """
            Calculate EI for a single point x_new.
            """
            x_new = np.array([x_new])
            mean, var = gp_predict(x_new, X, Y, L, alpha, l, sig, noise)
            mu = mean[0]
            sigma = np.sqrt(var[0])
            
            if sigma < 1e-8:
                return max(0, best_f_current - mu)
            
            z = (best_f_current - mu) / sigma
            ei = (best_f_current - mu) * norm.cdf(z) + sigma * norm.pdf(z)
            return ei

        # Optimization loop
        # We will use the remaining budget.
        # To speed up EI optimization, we use a small local search (Nelder-Mead) around promising points.
        
        while calls_used < total_budget and time.time() < deadline:
            # Fit GP to current data
            if len(Y_obs) < 2:
                # Not enough data, just random search
                x_new = rng.uniform(-5, 5, dim)
                evaluate(x_new)
                continue
                
            X = np.array(X_obs)
            Y = np.array(Y_obs)
            
            K, L, alpha = fit_gp(X, Y, length_scale, np.sqrt(signal_var), noise_var)
            
            if L is None:
                # GP fitting failed, fallback to random or Nelder-Mead on true func
                # Let's do a quick Nelder-Mead polish on the current best
                res = minimize(lambda x: evaluate(x), best_x, method='Nelder-Mead', 
                               options={'maxiter': 10, 'fatol': 1e-8})
                continue

            # Find next point by maximizing EI
            # We'll try a few random starting points for the EI optimizer and pick the best
            # This helps avoid local minima in the acquisition function
            
            best_ei_val = -np.inf
            best_ei_x = None
            
            # Generate a few candidate starting points for EI optimization
            # 1. Current best point
            # 2. Random points
            # 3. Points with high uncertainty (high variance)
            
            # Predict variance for a grid of points to find high uncertainty areas?
            # Too expensive. Instead, just use random starts and the best point.
            
            start_points = [best_x.copy()]
            for _ in range(5):
                start_points.append(rng.uniform(-5, 5, dim))
                
            for sp in start_points:
                # Optimize EI locally from sp
                # We want to maximize EI, so minimize -EI
                def neg_ei(x):
                    return -expected_improvement(x, X, Y, L, alpha, length_scale, np.sqrt(signal_var), noise_var, best_f)
                
                try:
                    res = minimize(neg_ei, sp, method='Nelder-Mead', 
                                   options={'maxiter': 50, 'xatol': 0.1, 'fatol': 1e-6})
                    if res.success or res.fun < -best_ei_val:
                        ei_val = -res.fun
                        if ei_val > best_ei_val:
                            best_ei_val = ei_val
                            best_ei_x = res.x
                except Exception:
                    continue
            
            if best_ei_x is not None:
                x_new = np.clip(best_ei_x, -5, 5)
                evaluate(x_new)
            else:
                # Fallback to random
                x_new = rng.uniform(-5, 5, dim)
                evaluate(x_new)
                
            # Check if we should switch to pure local optimization (Nelder-Mead)
            # If the GP uncertainty is very low everywhere, or if we are close to budget end
            # Simple heuristic: If best_f hasn't improved in the last 20 calls, do local polish
            if len(Y_obs) > 20:
                recent_20 = Y_obs[-20:]
                min_recent = np.min(recent_20)
                if min_recent >= best_f: # No improvement in last 20
                    # Switch to Nelder-Mead on true function for the remaining budget
                    # This is a structural change: from surrogate to direct local search
                    # Use Nelder-Mead to polish best_x
                    # We have (total_budget - calls_used) calls left
                    # Nelder-Mead uses 1 call per iteration roughly (actually 5-6 for simplex update in 5D)
                    # Let's just run a few iterations
                    
                    # To avoid infinite loops in minimize, we wrap the function to count calls
                    nm_calls = 0
                    def nm_func(x):
                        nonlocal nm_calls
                        if nm_calls >= (total_budget - calls_used) // 2: # Use half remaining for NM
                            return best_f
                        nm_calls += 1
                        return evaluate(x)
                    
                    try:
                        # Maximize number of iterations allowed
                        max_iter = max(10, (total_budget - calls_used) // 5)
                        res = minimize(nm_func, best_x, method='Nelder-Mead', 
                                       options={'maxiter': max_iter, 'fatol': 1e-10})
                    except Exception:
                        pass
                    break # Exit surrogate loop, we've used the budget for polishing
        
        return best_x, best_f
