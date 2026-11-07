"""
Bayesian Optimization with GP surrogate and EI acquisition for BBOB-style black-box optimization.

Strategy:
- Use a Gaussian Process surrogate model to learn the landscape from evaluations.
- Use Expected Improvement (EI) acquisition to balance exploration and exploitation.
- Initial phase: diverse sampling (Latin hypercube + center) to build initial model.
- Main phase: iteratively select next point via EI maximization using a dense grid
  or random candidates, evaluate, update GP.
- Final phase: local polish with L-BFGS-B if budget allows.

This is a structural shift from CMA-ES to model-based optimization, which should
find basins more efficiently on multimodal functions like Rastrigin and Gallagher.
"""
import numpy as np
import time
from scipy.optimize import minimize
from scipy.stats import norm
from math import erf, sqrt as msqrt


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # 10s safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        calls = [0]

        def wrapped(x):
            calls[0] += 1
            if calls[0] > budget:
                raise RuntimeError("budget exceeded")
            return float(np.clip(func(x), -1e15, 1e15))

        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        X_obs = []  # list of observed points
        y_obs = []  # list of observed values

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        def check_time():
            return time.time() > deadline

        # Phase 1: Initial diverse sampling using Latin Hypercube + center
        n_init = min(20, budget // 4)
        
        # Generate Latin Hypercube samples
        if n_init >= 2:
            lhc = np.zeros((n_init, dim))
            for d in range(dim):
                perm = rng.permutation(n_init)
                lhc[:, d] = (perm + 0.5) / n_init
            lhc = lhc * 10.0 - 5.0  # scale to [-5, 5]
        else:
            lhc = np.zeros((n_init, dim))

        # Add center point
        if n_init > 0:
            lhc[0] = np.zeros(dim)

        for i in range(n_init):
            if calls[0] >= budget or check_time():
                break
            x = lhc[i]
            f_val = wrapped(x)
            update_best(x, f_val)
            X_obs.append(x.copy())
            y_obs.append(f_val)

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped(best_x)
            X_obs.append(best_x.copy())
            y_obs.append(best_f)

        # GP surrogate model
        def rbf_kernel(X1, X2, length_scale=1.0, variance=1.0):
            """RBF kernel between two sets of points."""
            sq_dists = np.sum(X1 ** 2, axis=1)[:, np.newaxis] + \
                       np.sum(X2 ** 2, axis=1)[np.newaxis, :] - \
                       2 * X1 @ X2.T
            sq_dists = np.maximum(sq_dists, 0)
            return variance * np.exp(-0.5 * sq_dists / (length_scale ** 2))

        def gp_predict(X_new, X_train, y_train, length_scale=1.5, variance=1.0, noise=1e-6):
            """Gaussian Process prediction at X_new given training data."""
            K = rbf_kernel(X_train, X_train, length_scale, variance)
            K += noise * np.eye(len(X_train))
            K_star = rbf_kernel(X_train, X_new, length_scale, variance)
            try:
                K_inv_y = np.linalg.solve(K, y_train)
                mu = K_star.T @ K_inv_y
                K_ss = rbf_kernel(X_new, X_new, length_scale, variance)
                var = np.diag(K_ss) - np.sum(K_star.T * K_inv_y, axis=0)
                var = np.maximum(var, 1e-10)
                return mu, var
            except np.linalg.LinAlgError:
                # Fallback: use nearest neighbor
                dists = np.sum((X_new[:, np.newaxis, :] - X_train[np.newaxis, :, :]) ** 2, axis=2)
                nn_idx = np.argmin(dists, axis=1)
                mu = y_train[nn_idx]
                var = np.full(len(X_new), 1.0)
                return mu, var

        def expected_improvement(X_candidates, mu, var, best_val, xi=0.01):
            """Compute Expected Improvement for each candidate."""
            sigma = np.sqrt(var)
            imp = best_val - mu - xi
            with np.errstate(divide='ignore', invalid='ignore'):
                z = imp / sigma
                ei = imp * norm.cdf(z) + sigma * norm.pdf(z)
                ei[~np.isfinite(ei)] = 0.0
            return ei

        def find_next_point(X_obs, y_obs, best_f, n_candidates=2000):
            """Find the next point to evaluate using EI acquisition."""
            X_obs_arr = np.array(X_obs)
            y_obs_arr = np.array(y_obs)
            
            # Adapt length scale based on spread of data
            if len(X_obs_arr) > 1:
                spread = np.std(X_obs_arr, axis=0)
                spread = np.maximum(spread, 0.1)
                length_scale = np.mean(spread) * 0.8 + 0.5
            else:
                length_scale = 2.0
            
            # Generate candidate points: mix of random and grid-like
            candidates = []
            
            # Random candidates
            n_rand = n_candidates // 2
            rand_pts = rng.uniform(-5, 5, (n_rand, dim))
            candidates.append(rand_pts)
            
            # Points near current best
            n_near = n_candidates // 4
            near_sigma = 1.0
            near_pts = best_x + rng.standard_normal((n_near, dim)) * near_sigma
            near_pts = np.clip(near_pts, -5, 5)
            candidates.append(near_pts)
            
            # Points at intermediate distances
            n_mid = n_candidates - n_rand - n_near
            if n_mid > 0:
                mid_sigma = 2.5
                mid_pts = best_x + rng.standard_normal((n_mid, dim)) * mid_sigma
                mid_pts = np.clip(mid_pts, -5, 5)
                candidates.append(mid_pts)
            
            X_candidates = np.vstack(candidates)
            
            # GP prediction
            mu, var = gp_predict(X_candidates, X_obs_arr, y_obs_arr, 
                                  length_scale=length_scale, variance=1.0, noise=1e-6)
            
            # Compute EI
            ei = expected_improvement(X_candidates, mu, var, best_f, xi=0.01)
            
            # Select best candidate
            best_idx = np.argmax(ei)
            return X_candidates[best_idx]

        # Phase 2: Main Bayesian Optimization loop
        bo_iter = 0
        while calls[0] < budget - 5 and not check_time():
            bo_iter += 1
            
            # Find next point
            next_x = find_next_point(X_obs, y_obs, best_f, n_candidates=1500)
            next_x = np.clip(next_x, -5, 5)
            
            # Evaluate
            f_val = wrapped(next_x)
            update_best(next_x, f_val)
            X_obs.append(next_x.copy())
            y_obs.append(f_val)
            
            # Check if we should stop early if convergence is good
            if bo_iter > 50 and best_f < 1e-6:
                break

        # Phase 3: Final polish with L-BFGS-B if budget allows
        if best_x is not None and calls[0] < budget - 3 and not check_time():
            try:
                res = minimize(
                    wrapped,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 30, 'maxls': 10, 'gtol': 1e-8, 'ftol': 1e-12}
                )
                if res.fun < best_f:
                    update_best(res.x, res.fun)
            except Exception:
                pass

        # One more polish round if budget allows
        if best_x is not None and calls[0] < budget - 2 and not check_time():
            try:
                res = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 20, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res.fun < best_f:
                    update_best(res.x, res.fun)
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = wrapped(best_x)

        return best_x, best_f
