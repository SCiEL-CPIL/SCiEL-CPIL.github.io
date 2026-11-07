"""GP surrogate + LCB acquisition for BBOB f22 (Gallagher 21 peaks)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


class Optimizer:
    """Gaussian Process surrogate with LCB acquisition for global optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _gp_predict(self, X_train, y_train, X_test, length_scale, noise_var):
        """Predict mean and variance of GP at X_test given training data."""
        # RBF kernel: k(x, x') = exp(-||x-x'||^2 / (2 * l^2))
        D = cdist(X_train, X_test)  # (n_train, n_test)
        K = np.exp(-D**2 / (2.0 * length_scale**2))  # (n_train, n_test)
        
        # Add noise to diagonal for training covariance
        n_train = len(X_train)
        K_train = K.copy()
        K_train[np.arange(n_train), np.arange(n_train)] += noise_var
        
        try:
            L = np.linalg.cholesky(K_train + 1e-10 * np.eye(n_train))
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
        except np.linalg.LinAlgError:
            # Fallback: ridge regression
            try:
                alpha = np.linalg.solve(K_train + 1e-8 * np.eye(n_train), y_train)
            except np.linalg.LinAlgError:
                return np.full(len(X_test), np.mean(y_train)), np.full(len(X_test), np.var(y_train) + 1.0)
        
        # Predicted mean
        mu = K.T @ alpha
        # Predicted variance
        v = np.linalg.solve(L, K)
        var = 1.0 - np.sum(v**2, axis=0)
        var = np.maximum(var, 1e-12)
        
        return mu, var

    def _fit_gp_params(self, X, y, n_iter=30):
        """Fit GP hyperparameters via grid search on log-scale."""
        best_ll = -np.inf
        best_ls = 1.0
        best_nv = 1e-4
        
        # Grid of length scales and noise vars
        ls_candidates = np.logspace(-1.5, 1.5, 8)  # 0.03 to 31.6
        nv_candidates = np.logspace(-5, -1, 5)    # 1e-5 to 0.1
        
        # Subsample if too many points
        n = len(X)
        if n > 50:
            idx = np.random.default_rng(0).choice(n, 50, replace=False)
            X_sub, y_sub = X[idx], y[idx]
        else:
            X_sub, y_sub = X, y
        
        for ls in ls_candidates:
            for nv in nv_candidates:
                try:
                    D = cdist(X_sub, X_sub)
                    K = np.exp(-D**2 / (2.0 * ls**2)) + nv * np.eye(len(X_sub))
                    L = np.linalg.cholesky(K + 1e-10 * np.eye(len(X_sub)))
                    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_sub))
                    ll = -0.5 * (len(y_sub) * np.log(2 * np.pi) + 
                                 np.sum(np.log(np.diag(L)) * 2) + 
                                 y_sub @ alpha)
                    if ll > best_ll:
                        best_ll = ll
                        best_ls = ls
                        best_nv = nv
                except:
                    continue
        
        return best_ls, best_nv

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 280.0  # safety margin
        dim = self.dim
        budget = self.budget
        
        eval_count = 0
        best_x = None
        best_f = float('inf')
        X_all = []  # list of evaluated points
        y_all = []  # list of function values
        
        def wrapped_func(x):
            nonlocal eval_count, best_x, best_f
            if eval_count >= budget:
                return float('inf')
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            eval_count += 1
            f_val = float(func(x))
            X_all.append(x.copy())
            y_all.append(f_val)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val
        
        rng = np.random.default_rng(42)
        
        # Phase 1: Initial space-filling sampling (Sobol-like via Halton)
        def halton_sequence(n, dim, rng_seed=42):
            """Generate Halton sequence points in [0,1]^dim."""
            points = np.zeros((n, dim))
            for d in range(dim):
                base = d + 2  # primes: 2, 3, 5, 7, 11, ...
                for i in range(n):
                    f = 1.0
                    r = 0.0
                    idx = i + 1
                    while idx > 0:
                        f /= base
                        r += f * (idx % base)
                        idx //= base
                    points[i, d] = r
            # Shuffle for better mixing
            perm = rng.permutation(n)
            return points[perm]
        
        n_init = min(20, budget - 10)
        init_points = halton_sequence(n_init, dim, rng_seed=42)
        # Map from [0,1]^dim to [-5,5]^dim
        init_points = init_points * 10.0 - 5.0
        
        for pt in init_points:
            if eval_count >= budget - 5 or time.time() > deadline:
                break
            wrapped_func(pt)
        
        # Phase 2: GP-based active learning loop
        # Use LCB (Lower Confidence Bound) acquisition:
        # alpha(x) = mu(x) - kappa * sigma(x)
        # We want to minimize this
        
        gp_length_scale = 1.5
        gp_noise_var = 1e-3
        
        n_gp_iters = 0
        max_gp_iters = 400  # safety cap
        
        while eval_count < budget - 5 and time.time() < deadline and n_gp_iters < max_gp_iters:
            n_gp_iters += 1
            
            # Refit GP hyperparameters periodically
            if n_gp_iters % 10 == 1 and len(X_all) >= 10:
                try:
                    gp_length_scale, gp_noise_var = self._fit_gp_params(
                        np.array(X_all), np.array(y_all)
                    )
                except:
                    pass
            
            # Generate candidate points for acquisition
            # Mix of: random candidates, points near current best, and structured candidates
            n_candidates = 500
            
            candidates = []
            # Random candidates
            candidates.append(rng.uniform(-5.0, 5.0, (n_candidates // 2, dim)))
            # Points near current best
            if best_x is not None:
                near_best = best_x[None, :] + rng.normal(0, 1.0, (n_candidates // 4, dim))
                candidates.append(np.clip(near_best, -5.0, 5.0))
            # Structured candidates (Halton in remaining space)
            h = halton_sequence(n_candidates // 4, dim, rng_seed=42 + n_gp_iters)
            candidates.append(h * 10.0 - 5.0)
            
            candidates = np.vstack(candidates)
            candidates = np.clip(candidates, -5.0, 5.0)
            
            # Predict mean and variance at candidates
            X_train = np.array(X_all)
            y_train = np.array(y_all)
            
            try:
                mu, var = self._gp_predict(X_train, y_train, candidates, gp_length_scale, gp_noise_var)
                sigma = np.sqrt(var)
            except:
                # Fallback: just use random
                x_next = rng.uniform(-5.0, 5.0, dim)
                wrapped_func(x_next)
                continue
            
            # LCB acquisition: minimize mu - kappa * sigma
            # Adapt kappa based on progress
            n_evals = eval_count
            if n_evals < 30:
                kappa = 2.0  # more exploration early
            elif n_evals < 100:
                kappa = 1.5
            elif n_evals < 250:
                kappa = 1.0
            else:
                kappa = 0.5  # more exploitation late
            
            lcb = mu - kappa * sigma
            x_next = candidates[np.argmin(lcb)]
            x_next = np.clip(x_next, -5.0, 5.0)
            
            # Check if we're too close to an existing point
            if len(X_all) > 0:
                min_dist = np.min(np.linalg.norm(np.array(X_all) - x_next, axis=1))
                if min_dist < 1e-3:
                    # Pick second best
                    lcb_copy = lcb.copy()
                    lcb_copy[np.argmin(lcb)] = np.inf
                    x_next = candidates[np.argmin(lcb_copy)]
                    x_next = np.clip(x_next, -5.0, 5.0)
            
            wrapped_func(x_next)
        
        # Phase 3: Local refinement with L-BFGS-B from current best
        if eval_count < budget - 3 and time.time() < deadline and best_x is not None:
            try:
                minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 30, 'maxls': 15, 'ftol': 1e-12, 'gtol': 1e-8}
                )
            except Exception:
                pass
        
        # Phase 4: Fill remaining budget with evaluations near best
        while eval_count < budget and time.time() < deadline and best_x is not None:
            x = best_x + rng.normal(0, 0.05, dim)
            x = np.clip(x, -5.0, 5.0)
            wrapped_func(x)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
