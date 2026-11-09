"""Surrogate-assisted Bayesian Optimization with Gaussian Process for BBOB f21."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

class Optimizer:
    """Surrogate-assisted Bayesian Optimization with Gaussian Process for BBOB f21."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        calls_made = 0
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        X_obs = []
        y_obs = []

        def count_call():
            nonlocal calls_made
            calls_made += 1

        def evaluate(x):
            nonlocal best_x, best_f, X_obs, y_obs
            x = np.clip(x, -5, 5)
            f = float(func(x))
            count_call()
            if f < best_f:
                best_f = f
                best_x = x.copy()
            X_obs.append(x)
            y_obs.append(f)
            return f

        # Phase 1: Initialization
        # Use a mix of random and Halton-like points to cover the space
        n_init = 10
        init_points = []
        # Simple random start
        init_points.append(rng.uniform(-5, 5, dim))
        # Halton-like for diversity
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r
        
        bases = [2, 3, 5, 7, 11]
        for i in range(1, n_init):
            point = np.array([halton(i, bases[d]) for d in range(dim)])
            point = point * 10.0 - 5.0
            init_points.append(point)
        
        for p in init_points:
            if time.time() > deadline or calls_made >= budget:
                break
            evaluate(p)

        if not X_obs:
            return best_x, best_f

        X_obs = np.array(X_obs)
        y_obs = np.array(y_obs)
        
        # Normalize y for GP stability
        y_mean = np.mean(y_obs)
        y_std = np.std(y_obs)
        if y_std < 1e-8:
            y_std = 1.0
        y_norm = (y_obs - y_mean) / y_std

        # GP Hyperparameters
        # Length-scale l, Signal variance sigma_f, Noise variance sigma_n
        l = 1.0
        sigma_f = 1.0
        sigma_n = 1e-4

        def rbf_kernel(X1, X2, l):
            sq_dists = np.sum(X1**2, axis=1)[:, np.newaxis] + np.sum(X2**2, axis=1)[np.newaxis, :] - 2 * X1 @ X2.T
            sq_dists = np.maximum(sq_dists, 0)
            return sigma_f**2 * np.exp(-0.5 * sq_dists / l**2)

        def fit_gp(X, y, l, sigma_f, sigma_n):
            K = rbf_kernel(X, X, l)
            K += sigma_n**2 * np.eye(len(X))
            try:
                L = np.linalg.cholesky(K)
                alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                return L, alpha
            except np.linalg.LinAlgError:
                # Add jitter if Cholesky fails
                K += 1e-6 * np.eye(len(X))
                try:
                    L = np.linalg.cholesky(K)
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                    return L, alpha
                except np.linalg.LinAlgError:
                    return None, None

        def predict_gp(X_new, L, alpha, X_obs, y_obs, l, sigma_f, sigma_n):
            K_star = rbf_kernel(X_obs, X_new, l)
            K_new = rbf_kernel(X_new, X_new, l)
            
            # Mean
            K_star_T = K_star.T
            mean = K_star_T @ alpha
            
            # Variance
            v = np.linalg.solve(L, K_star)
            var = np.diag(K_new) - np.sum(v**2, axis=0)
            var = np.maximum(var, 1e-10)
            
            return mean, np.sqrt(var)

        # Optimize hyperparameters every K steps
        def optimize_hyperparams(X, y, l_init, sigma_f_init, sigma_n_init):
            # Simple grid or random search for l and sigma_f
            best_log_marg = -np.inf
            best_params = (l_init, sigma_f_init, sigma_n_init)
            
            # Keep sigma_n fixed small
            candidates_l = [0.1, 0.5, 1.0, 2.0, 5.0]
            candidates_sf = [0.1, 0.5, 1.0, 2.0]
            
            for l_c in candidates_l:
                for sf_c in candidates_sf:
                    K = rbf_kernel(X, X, l_c)
                    K += sigma_n_init**2 * np.eye(len(X))
                    try:
                        L = np.linalg.cholesky(K)
                        alpha = np.linalg.solve(L.T, np.linalg.solve(L, y))
                        log_marg = -0.5 * np.dot(y, alpha) - 0.5 * np.sum(np.log(np.diag(L)) * 2) - 0.5 * len(y) * np.log(2 * np.pi)
                        if log_marg > best_log_marg:
                            best_log_marg = log_marg
                            best_params = (l_c, sf_c, sigma_n_init)
                    except:
                        continue
            return best_params

        L, alpha = fit_gp(X_obs, y_norm, l, sigma_f, sigma_n)
        
        current_beta = 5.0
        min_beta = 0.5
        beta_decay = 0.995
        
        while calls_made < budget:
            if time.time() > deadline:
                break
                
            # Optimize hyperparameters every 10 steps
            if len(X_obs) % 10 == 0 and len(X_obs) > 10:
                l, sigma_f, sigma_n = optimize_hyperparams(X_obs, y_norm, l, sigma_f, sigma_n)
                L, alpha = fit_gp(X_obs, y_norm, l, sigma_f, sigma_n)
                if L is None:
                    break

            # Define acquisition function (UCB)
            def acquisition(x):
                x = x.reshape(1, -1)
                mean, std = predict_gp(x, L, alpha, X_obs, y_norm, l, sigma_f, sigma_n)
                # UCB: mean + beta * std
                # Note: y_norm is normalized. We need to be careful with scaling.
                # Since we just want to maximize the acquisition, and y_norm is linear transform of y,
                # maximizing UCB on y_norm is equivalent to maximizing on y if beta is scaled appropriately.
                # However, the 'std' is in normalized units.
                # Let's stick to normalized space for the GP.
                ucb = mean[0] + current_beta * std[0]
                return -ucb # Minimize negative UCB

            # Maximize acquisition using multi-start Nelder-Mead
            best_acq_val = np.inf
            best_x_next = None
            
            # Starts: current best, random, and points near best
            starts = [best_x.copy()]
            for _ in range(10):
                starts.append(rng.uniform(-5, 5, dim))
            for _ in range(5):
                starts.append(best_x + rng.standard_normal(dim) * 0.5)
                
            for x0 in starts:
                if time.time() > deadline or calls_made >= budget:
                    break
                res = minimize(acquisition, x0, method='Nelder-Mead', 
                               options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6})
                if res.fun < best_acq_val:
                    best_acq_val = res.fun
                    best_x_next = res.x
            
            if best_x_next is None:
                best_x_next = rng.uniform(-5, 5, dim)
            
            best_x_next = np.clip(best_x_next, -5, 5)
            
            # Evaluate
            f_new = evaluate(best_x_next)
            
            # Update GP data
            X_obs = np.vstack([X_obs, best_x_next])
            y_new = (f_new - y_mean) / y_std
            y_obs = np.concatenate([y_obs, [y_new]])
            y_norm = y_obs
            
            # Update beta (annealing)
            current_beta = max(min_beta, current_beta * beta_decay)
            
            # Re-fit GP with new point (fast, just update Cholesky or refit)
            # Refitting is safer and fast enough for N<500
            L, alpha = fit_gp(X_obs, y_norm, l, sigma_f, sigma_n)
            if L is None:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
